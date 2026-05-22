"""Postgres integration test for the fm_no_lookahead trigger (migration 004).

The Python-side `FeatureMatrixValidator` is unit-tested elsewhere; this file
tests the DB constraint that is the LAST line of defense if the validator
is bypassed (e.g. a future raw-SQL writer or a buggy ORM call). The trigger
must reject every variant of as_of_ts >= match boundary for both live and
historical rows.

Skips cleanly when Docker / testcontainers / Postgres deps are unavailable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest


# Minimal player/tournament/venue rows just so matches FK can be satisfied.
def _seed_match(conn: Any, *, match_id: int, start_ts: datetime | None,
                match_date: date) -> None:
    from sqlalchemy import text

    # Idempotent seeds — repeated calls reuse the same player / tournament.
    conn.execute(text("""
        INSERT INTO venues (venue_id, city, country_code)
        VALUES (1, 'Test City', 'TST')
        ON CONFLICT (venue_id) DO NOTHING;
    """))
    conn.execute(text("""
        INSERT INTO tournaments (
            tournament_id, season, slug, name, tier, surface, indoor, venue_id
        )
        VALUES (1, 2026, 'test-slug', 'Test Open', 'GS', 'Hard', false, 1)
        ON CONFLICT (tournament_id) DO NOTHING;
    """))
    conn.execute(text("""
        INSERT INTO players (player_id, full_name, source, source_uid)
        VALUES
            (100, 'Player One', 'test', 'p1'),
            (200, 'Player Two', 'test', 'p2')
        ON CONFLICT (player_id) DO NOTHING;
    """))
    conn.execute(text("""
        INSERT INTO matches (
            match_id, tournament_id, round, match_date, start_ts,
            p1_id, p2_id, status, source, source_uid
        )
        VALUES (
            :mid, 1, 'QF', :md, :sts,
            100, 200, 'scheduled', 'test', :uid
        )
        ON CONFLICT (match_id) DO UPDATE
          SET match_date = EXCLUDED.match_date,
              start_ts   = EXCLUDED.start_ts;
    """), {"mid": match_id, "md": match_date, "sts": start_ts, "uid": f"m{match_id}"})


def _insert_feature(conn: Any, *, match_id: int, as_of: datetime,
                    feature_set: str = "v1") -> None:
    from sqlalchemy import text
    conn.execute(text("""
        INSERT INTO feature_matrix (match_id, feature_set, as_of_ts, payload)
        VALUES (:mid, :fs, :as_of, '{}'::jsonb);
    """), {"mid": match_id, "fs": feature_set, "as_of": as_of})


@pytest.fixture
def conn(postgres_engine: Any) -> Any:
    """A connection per test, rolled back at the end so tests don't bleed."""
    with postgres_engine.connect() as c:
        tx = c.begin()
        try:
            yield c
        finally:
            tx.rollback()


class TestLiveMatchPIT:
    """matches.start_ts IS NOT NULL → as_of_ts < start_ts is enforced."""

    START_TS = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
    MATCH_DATE = date(2026, 5, 21)

    def test_as_of_well_before_start_ts_accepted(self, conn: Any) -> None:
        _seed_match(conn, match_id=9001, start_ts=self.START_TS,
                    match_date=self.MATCH_DATE)
        _insert_feature(conn, match_id=9001,
                        as_of=self.START_TS - timedelta(days=1))

    def test_as_of_equal_to_start_ts_rejected(self, conn: Any) -> None:
        from sqlalchemy.exc import InternalError, ProgrammingError
        _seed_match(conn, match_id=9002, start_ts=self.START_TS,
                    match_date=self.MATCH_DATE)
        with pytest.raises((InternalError, ProgrammingError), match="lookahead"):
            _insert_feature(conn, match_id=9002, as_of=self.START_TS)

    def test_as_of_after_start_ts_rejected(self, conn: Any) -> None:
        from sqlalchemy.exc import InternalError, ProgrammingError
        _seed_match(conn, match_id=9003, start_ts=self.START_TS,
                    match_date=self.MATCH_DATE)
        with pytest.raises((InternalError, ProgrammingError), match="lookahead"):
            _insert_feature(conn, match_id=9003,
                            as_of=self.START_TS + timedelta(minutes=1))


class TestHistoricalMatchPIT:
    """matches.start_ts IS NULL → as_of_ts < match_date(UTC midnight) enforced."""

    MATCH_DATE = date(2020, 6, 15)
    MIDNIGHT_UTC = datetime(2020, 6, 15, 0, 0, tzinfo=UTC)

    def test_as_of_day_before_accepted(self, conn: Any) -> None:
        _seed_match(conn, match_id=9101, start_ts=None,
                    match_date=self.MATCH_DATE)
        _insert_feature(conn, match_id=9101,
                        as_of=self.MIDNIGHT_UTC - timedelta(days=1))

    def test_as_of_equal_to_midnight_utc_rejected(self, conn: Any) -> None:
        from sqlalchemy.exc import InternalError, ProgrammingError
        _seed_match(conn, match_id=9102, start_ts=None,
                    match_date=self.MATCH_DATE)
        with pytest.raises((InternalError, ProgrammingError), match="lookahead"):
            _insert_feature(conn, match_id=9102, as_of=self.MIDNIGHT_UTC)

    def test_as_of_after_match_date_rejected(self, conn: Any) -> None:
        from sqlalchemy.exc import InternalError, ProgrammingError
        _seed_match(conn, match_id=9103, start_ts=None,
                    match_date=self.MATCH_DATE)
        with pytest.raises((InternalError, ProgrammingError), match="lookahead"):
            _insert_feature(conn, match_id=9103,
                            as_of=self.MIDNIGHT_UTC + timedelta(hours=1))


class TestTriggerInvariants:
    def test_unknown_match_id_rejected(self, conn: Any) -> None:
        # No match seeded for id=9999 — FK + trigger should both refuse.
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            _insert_feature(conn, match_id=9999,
                            as_of=datetime(2026, 1, 1, tzinfo=UTC))

    def test_trigger_fires_on_update_not_just_insert(self, conn: Any) -> None:
        """An UPDATE that walks as_of_ts past start_ts must be rejected."""
        from sqlalchemy import text
        from sqlalchemy.exc import InternalError, ProgrammingError

        start = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        _seed_match(conn, match_id=9200, start_ts=start,
                    match_date=date(2026, 5, 21))
        _insert_feature(conn, match_id=9200,
                        as_of=start - timedelta(hours=24))

        with pytest.raises((InternalError, ProgrammingError), match="lookahead"):
            conn.execute(text("""
                UPDATE feature_matrix
                   SET as_of_ts = :bad
                 WHERE match_id = 9200 AND feature_set = 'v1';
            """), {"bad": start + timedelta(minutes=1)})

    def test_session_tz_non_utc_does_not_loosen_trigger(self, conn: Any) -> None:
        """The trigger's explicit `match_date::timestamp AT TIME ZONE 'UTC'`
        must not be loosened when the session TZ is non-UTC. Regression
        test for the H1 fix (defense-in-depth against future ALTER SESSION).
        """
        from sqlalchemy import text
        from sqlalchemy.exc import InternalError, ProgrammingError

        conn.execute(text("SET TIME ZONE 'America/New_York'"))
        match_date = date(2020, 6, 15)
        _seed_match(conn, match_id=9300, start_ts=None, match_date=match_date)

        # 00:00 UTC on match_date is the trigger boundary regardless of
        # session TZ. In NY local time that's 20:00 the previous day.
        boundary = datetime(2020, 6, 15, 0, 0, tzinfo=UTC)
        with pytest.raises((InternalError, ProgrammingError), match="lookahead"):
            _insert_feature(conn, match_id=9300, as_of=boundary)
