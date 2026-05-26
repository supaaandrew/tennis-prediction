"""Integration tests for concrete repositories.

These tests run the FULL upsert + query lifecycle against a real
Postgres instance (testcontainers). They auto-skip when Docker isn't
available — same pattern as `tests/integration/test_pit_trigger.py`.

What's covered here that the unit tests can't:
  - `for_training()` returns only `status='final'` rows (SQL filter).
  - `for_prediction()` returns only `scheduled`/`live` rows (SQL filter).
  - `WeatherObservationRepositoryImpl.upsert()` writes a
    `weather_revisions` row on overwrite (audit trail).
  - Upsert idempotency: same natural key + two calls = one row.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from tennis.core.ids import (
    match_id as compute_match_id,
    player_id_from_source,
    tournament_id as compute_tournament_id,
    venue_id as compute_venue_id,
)


# ---------------------------------------------------------------------------
# Session factory bound to the testcontainers engine (from conftest)
# ---------------------------------------------------------------------------
@pytest.fixture
def session_factory(postgres_engine: Any):
    """Yields a callable matching the SessionCallable contract.

    We don't use PostgresSessionFactory here because that creates its own
    engine; the testcontainers fixture already provides one we want to
    reuse so all tests share the same migrated database.
    """
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(postgres_engine, expire_on_commit=False)

    @contextmanager
    def _factory():
        s = Session()
        # Mirror PostgresSessionFactory's UTC enforcement on each session.
        s.execute(text("SET TIME ZONE 'UTC'"))
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return _factory


# ---------------------------------------------------------------------------
# Seed helpers — minimal entity hierarchy so FKs are satisfied
# ---------------------------------------------------------------------------
def _seed_player(repo, *, source_uid: str, name: str = "Test Player"):
    from tennis.storage.postgres.rows import PlayerRow

    row = PlayerRow(
        player_id=player_id_from_source(source="test", source_uid=source_uid),
        full_name=name,
        source="test",
        source_uid=source_uid,
    )
    return repo.upsert(row)


def _seed_venue(repo, *, city: str = "Test City", country: str = "TST"):
    from tennis.storage.postgres.rows import VenueRow

    row = VenueRow(
        venue_id=compute_venue_id(city=city, country_code=country),
        city=city,
        country_code=country,
    )
    return repo.upsert(row)


def _seed_tournament(repo, *, slug: str, tier: str = "GS",
                     season: int = 2026, surface: str = "Hard",
                     venue_id: int | None = None):
    from tennis.storage.postgres.rows import TournamentRow

    row = TournamentRow(
        tournament_id=compute_tournament_id(season=season, slug=slug),
        season=season,
        slug=slug,
        name=slug.replace("-", " ").title(),
        tier=tier,
        surface=surface,
        indoor=False,
        venue_id=venue_id,
    )
    return repo.upsert(row)


def _seed_match(repo, *, tournament_id: int, p1_id: int, p2_id: int,
                round: str, match_date: date, status: str = "final",
                source_uid: str | None = None,
                start_ts: datetime | None = None):
    from tennis.storage.postgres.rows import MatchRow

    row = MatchRow(
        match_id=compute_match_id(
            tournament_id=tournament_id, round=round,
            player_a=p1_id, player_b=p2_id, match_date=match_date,
        ),
        tournament_id=tournament_id,
        round=round,
        match_date=match_date,
        p1_id=p1_id,
        p2_id=p2_id,
        status=status,
        source="test",
        source_uid=source_uid or f"m-{tournament_id}-{round}-{p1_id}-{p2_id}",
        start_ts=start_ts,
        match_date_source="sackmann",
    )
    return repo.upsert(row)


# ---------------------------------------------------------------------------
# 1. for_training() — only status='final', only allowed tiers
# ---------------------------------------------------------------------------
class TestForTraining:
    def test_returns_only_final_status(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="ForTrainCity1", country="T01")
        t = _seed_tournament(tournaments, slug="train-1", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="ftA")
        p2 = _seed_player(players, source_uid="ftB")

        _seed_match(matches, tournament_id=t.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="QF", match_date=date(2026, 1, 25),
                    status="final", source_uid="ft-final")
        _seed_match(matches, tournament_id=t.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="SF", match_date=date(2026, 1, 26),
                    status="scheduled", source_uid="ft-sched")
        _seed_match(matches, tournament_id=t.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="F", match_date=date(2026, 1, 27),
                    status="cancelled", source_uid="ft-cancelled")

        rows = list(matches.for_training(season_start=2026, season_end=2026))
        statuses = {r.status for r in rows}
        # Only finals — never scheduled, live, or cancelled.
        assert statuses == {"final"}, f"got statuses: {statuses}"

    def test_excludes_disallowed_tiers(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="ForTrainCity2", country="T02")
        # Two tournaments — one allowed tier, one excluded.
        t_gs = _seed_tournament(tournaments, slug="train-2-gs",
                                tier="GS", venue_id=v.venue_id)
        t_chl = _seed_tournament(tournaments, slug="train-2-chl",
                                 tier="Challenger", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="ft2A")
        p2 = _seed_player(players, source_uid="ft2B")

        _seed_match(matches, tournament_id=t_gs.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="QF", match_date=date(2026, 2, 1),
                    status="final", source_uid="ft2-gs")
        _seed_match(matches, tournament_id=t_chl.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="F", match_date=date(2026, 2, 2),
                    status="final", source_uid="ft2-chl")

        rows = list(matches.for_training(season_start=2026, season_end=2026))
        seen_tournaments = {r.tournament_id for r in rows}
        assert t_gs.tournament_id in seen_tournaments
        assert t_chl.tournament_id not in seen_tournaments


# ---------------------------------------------------------------------------
# 2. for_prediction() — only scheduled/live, no finals
# ---------------------------------------------------------------------------
class TestForPrediction:
    def test_returns_only_scheduled_and_live(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="ForPredCity1", country="P01")
        t = _seed_tournament(tournaments, slug="pred-1", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="fpA")
        p2 = _seed_player(players, source_uid="fpB")

        anchor_ts = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

        _seed_match(matches, tournament_id=t.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="QF", match_date=date(2026, 6, 2),
                    start_ts=anchor_ts + timedelta(hours=20),
                    status="scheduled", source_uid="fp-sched")
        _seed_match(matches, tournament_id=t.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="SF", match_date=date(2026, 6, 2),
                    start_ts=anchor_ts + timedelta(hours=22),
                    status="live", source_uid="fp-live")
        _seed_match(matches, tournament_id=t.tournament_id,
                    p1_id=p1.player_id, p2_id=p2.player_id,
                    round="F", match_date=date(2026, 6, 2),
                    start_ts=anchor_ts + timedelta(hours=23),
                    status="final", source_uid="fp-final")

        rows = list(matches.for_prediction(as_of=anchor_ts, lookforward_days=3))
        statuses = {r.status for r in rows}
        assert statuses.issubset({"scheduled", "live"})
        assert "final" not in statuses


# ---------------------------------------------------------------------------
# 3. WeatherObservationRepositoryImpl writes weather_revisions on overwrite
# ---------------------------------------------------------------------------
class TestWeatherAuditTrail:
    def test_overwrite_writes_to_weather_revisions(self, session_factory: Any) -> None:
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            VenueRepositoryImpl,
            WeatherObservationRepositoryImpl,
        )
        from tennis.storage.postgres.rows import WeatherObservationRow

        venues = VenueRepositoryImpl(session_factory)
        weather = WeatherObservationRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="WeatherCity", country="WX1")
        obs_ts = datetime(2026, 6, 10, 14, 0, tzinfo=UTC)

        first = WeatherObservationRow(
            venue_id=v.venue_id, observed_at=obs_ts, source="owm",
            is_forecast=False, temp_c=22.5, humidity_pct=55.0,
        )
        weather.upsert(first)

        # No revisions row yet.
        with session_factory() as s:
            count = s.execute(
                select(m.WeatherRevision).where(
                    m.WeatherRevision.venue_id == v.venue_id,
                    m.WeatherRevision.observed_at == obs_ts,
                )
            ).all()
            assert len(count) == 0

        # Overwrite with a new value -> must produce a revision row.
        second = WeatherObservationRow(
            venue_id=v.venue_id, observed_at=obs_ts, source="owm",
            is_forecast=False, temp_c=24.0, humidity_pct=60.0,
        )
        weather.upsert(second)

        with session_factory() as s:
            revs = s.execute(
                select(m.WeatherRevision).where(
                    m.WeatherRevision.venue_id == v.venue_id,
                    m.WeatherRevision.observed_at == obs_ts,
                )
            ).scalars().all()
            assert len(revs) == 1
            rev = revs[0]
            # previous_row carries the FIRST temp_c (22.5).
            assert rev.previous_row.get("temp_c") == 22.5
            assert rev.new_row.get("temp_c") == 24.0

    def test_revision_captures_first_value_as_previous_row(
        self, session_factory: Any
    ) -> None:
        """Audit-fix Test A — revision is written in the same statement
        as the overwrite. Calling upsert twice with different values
        produces exactly one revision row whose `previous_row` matches
        the FIRST upsert's values.
        """
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            VenueRepositoryImpl,
            WeatherObservationRepositoryImpl,
        )
        from tennis.storage.postgres.rows import WeatherObservationRow

        venues = VenueRepositoryImpl(session_factory)
        weather = WeatherObservationRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="AuditOrderCity", country="AO1")
        obs_ts = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)

        weather.upsert(WeatherObservationRow(
            venue_id=v.venue_id, observed_at=obs_ts, source="owm",
            is_forecast=False, temp_c=15.0, humidity_pct=50.0,
        ))
        weather.upsert(WeatherObservationRow(
            venue_id=v.venue_id, observed_at=obs_ts, source="owm",
            is_forecast=False, temp_c=21.0, humidity_pct=70.0,
        ))

        with session_factory() as s:
            revs = s.execute(
                select(m.WeatherRevision).where(
                    m.WeatherRevision.venue_id == v.venue_id,
                    m.WeatherRevision.observed_at == obs_ts,
                )
            ).scalars().all()
            assert len(revs) == 1
            # previous_row pins the FIRST upsert (15.0), not the second.
            assert revs[0].previous_row.get("temp_c") == 15.0
            assert revs[0].previous_row.get("humidity_pct") == 50.0
            # new_row pins the second upsert.
            assert revs[0].new_row.get("temp_c") == 21.0

    def test_overwrite_aborts_when_revision_insert_fails(
        self, session_factory: Any
    ) -> None:
        """Audit-fix Test B — the revision insert and the row update
        are in a SINGLE statement, so if the revision insert fails the
        overwrite aborts and the stored observation retains its
        original value. Simulated by installing a temporary BEFORE
        INSERT trigger on `weather_revisions` that raises.
        """
        from sqlalchemy import select, text

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            VenueRepositoryImpl,
            WeatherObservationRepositoryImpl,
        )
        from tennis.storage.postgres.rows import WeatherObservationRow

        venues = VenueRepositoryImpl(session_factory)
        weather = WeatherObservationRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="AuditAtomicCity", country="AT1")
        obs_ts = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)

        # Plant an original observation.
        weather.upsert(WeatherObservationRow(
            venue_id=v.venue_id, observed_at=obs_ts, source="owm",
            is_forecast=False, temp_c=10.0, humidity_pct=40.0,
        ))

        # Install a trigger that aborts every INSERT into weather_revisions.
        with session_factory() as s:
            s.execute(text(
                """
                CREATE OR REPLACE FUNCTION _atomic_block_revision()
                  RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION 'simulated revision-insert failure';
                END;
                $$ LANGUAGE plpgsql;
                """
            ))
            s.execute(text(
                "CREATE TRIGGER _atomic_block_revision_trg "
                "BEFORE INSERT ON weather_revisions "
                "FOR EACH ROW EXECUTE FUNCTION _atomic_block_revision();"
            ))

        try:
            # Attempt overwrite — must raise because the revision INSERT
            # in the CTE fails, aborting the whole statement.
            with pytest.raises(Exception, match="simulated revision-insert failure"):
                weather.upsert(WeatherObservationRow(
                    venue_id=v.venue_id, observed_at=obs_ts, source="owm",
                    is_forecast=False, temp_c=99.0, humidity_pct=99.0,
                ))

            # The observation row must still carry the ORIGINAL value
            # — the overwrite did not commit.
            with session_factory() as s:
                o = s.execute(
                    select(m.WeatherObservation).where(
                        m.WeatherObservation.venue_id == v.venue_id,
                        m.WeatherObservation.observed_at == obs_ts,
                    )
                ).scalar_one()
                assert o.temp_c == 10.0
                assert o.humidity_pct == 40.0

            # And no revision row was written (the trigger blocked it).
            with session_factory() as s:
                rev_count = s.execute(
                    select(m.WeatherRevision).where(
                        m.WeatherRevision.venue_id == v.venue_id,
                        m.WeatherRevision.observed_at == obs_ts,
                    )
                ).all()
                assert len(rev_count) == 0
        finally:
            # Clean up the trigger so other tests aren't affected.
            with session_factory() as s:
                s.execute(text(
                    "DROP TRIGGER IF EXISTS _atomic_block_revision_trg "
                    "ON weather_revisions;"
                ))
                s.execute(text(
                    "DROP FUNCTION IF EXISTS _atomic_block_revision();"
                ))

    def test_first_insert_creates_no_revision(self, session_factory: Any) -> None:
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            VenueRepositoryImpl,
            WeatherObservationRepositoryImpl,
        )
        from tennis.storage.postgres.rows import WeatherObservationRow

        venues = VenueRepositoryImpl(session_factory)
        weather = WeatherObservationRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="FirstInsCity", country="WX2")
        obs_ts = datetime(2026, 7, 4, 14, 0, tzinfo=UTC)
        weather.upsert(
            WeatherObservationRow(
                venue_id=v.venue_id, observed_at=obs_ts, source="owm",
                is_forecast=True, temp_c=18.0,
            )
        )
        with session_factory() as s:
            revs = s.execute(
                select(m.WeatherRevision).where(
                    m.WeatherRevision.venue_id == v.venue_id,
                    m.WeatherRevision.observed_at == obs_ts,
                )
            ).all()
            assert len(revs) == 0


# ---------------------------------------------------------------------------
# 4. Upsert idempotency — natural key dedup, two calls = one row
# ---------------------------------------------------------------------------
class TestUpsertIdempotency:
    def test_player_upsert_idempotent(self, session_factory: Any) -> None:
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import PlayerRepositoryImpl
        from tennis.storage.postgres.rows import PlayerRow

        players = PlayerRepositoryImpl(session_factory)
        pid = player_id_from_source(source="test", source_uid="idem-player")

        for full_name in ("Alpha One", "Alpha Two", "Alpha Three"):
            players.upsert(
                PlayerRow(
                    player_id=pid, full_name=full_name,
                    source="test", source_uid="idem-player",
                )
            )

        with session_factory() as s:
            rows = s.execute(
                select(m.Player).where(
                    m.Player.source == "test",
                    m.Player.source_uid == "idem-player",
                )
            ).scalars().all()
            assert len(rows) == 1
            assert rows[0].full_name == "Alpha Three"

    def test_match_upsert_idempotent(self, session_factory: Any) -> None:
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="IdemCity", country="ID1")
        t = _seed_tournament(tournaments, slug="idem-tourney", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="idemMatchA")
        p2 = _seed_player(players, source_uid="idemMatchB")

        # Three upserts with the same source_uid (natural key).
        for status in ("scheduled", "live", "final"):
            _seed_match(matches, tournament_id=t.tournament_id,
                        p1_id=p1.player_id, p2_id=p2.player_id,
                        round="F", match_date=date(2026, 8, 1),
                        status=status, source_uid="idem-match-uid")

        with session_factory() as s:
            rows = s.execute(
                select(m.Match).where(
                    m.Match.source == "test",
                    m.Match.source_uid == "idem-match-uid",
                )
            ).scalars().all()
            assert len(rows) == 1
            # Final write wins.
            assert rows[0].status == "final"

    def test_cross_source_upsert_merges_on_match_id_pk(self, session_factory: Any) -> None:
        """K4: a scraper row then a Sackmann row with the SAME match_id but a
        DIFFERENT source_uid must merge on the PK (no IntegrityError), promote
        to final, and PRESERVE the scraper's start_ts via COALESCE. Identity
        (source/source_uid) is kept from the first writer, not rewritten."""
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )
        from tennis.storage.postgres.rows import MatchRow

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="MergeCity", country="MG1")
        t = _seed_tournament(tournaments, slug="merge-t", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="mergeA")
        p2 = _seed_player(players, source_uid="mergeB")

        mid = compute_match_id(
            tournament_id=t.tournament_id, round="QF",
            player_a=p1.player_id, player_b=p2.player_id,
            match_date=date(2026, 8, 1),
        )
        scraper_ts = datetime(2026, 8, 1, 13, 30, tzinfo=UTC)

        # 1. Scraper writes the upcoming match first (distinct source_uid, K2).
        matches.upsert(MatchRow(
            match_id=mid, tournament_id=t.tournament_id, round="QF",
            match_date=date(2026, 8, 1), p1_id=p1.player_id, p2_id=p2.player_id,
            status="scheduled", source="atp_scraper",
            source_uid="merge-t:2026:QF:a:b", start_ts=scraper_ts,
            match_date_source="atp_scraper",
        ))

        # 2. Sackmann later publishes the SAME match: same match_id, different
        #    source_uid, status=final, start_ts NULL. Must NOT raise.
        merged = matches.upsert(MatchRow(
            match_id=mid, tournament_id=t.tournament_id, round="QF",
            match_date=date(2026, 8, 1), p1_id=p1.player_id, p2_id=p2.player_id,
            status="final", source="sackmann", source_uid="2026-540:1",
            start_ts=None, match_date_source="sackmann",
        ))

        assert merged.status == "final"          # final write wins (status mutable)
        assert merged.source == "atp_scraper"     # identity kept from FIRST writer (FIX3)
        assert merged.start_ts == scraper_ts      # COALESCE preserved start_ts

        with session_factory() as s:
            rows = s.execute(
                select(m.Match).where(m.Match.match_id == mid)
            ).scalars().all()
            assert len(rows) == 1                  # merged, not duplicated

    def test_update_live_fields_updates_only_scraper_owned(
        self, session_factory: Any
    ) -> None:
        """K1: update_live_fields touches only start_ts/status/match_date_source
        and no-ops (no raise, no insert) when the match_id is absent."""
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="LiveCity", country="LV1")
        t = _seed_tournament(tournaments, slug="live-t", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="liveA")
        p2 = _seed_player(players, source_uid="liveB")
        seeded = _seed_match(
            matches, tournament_id=t.tournament_id,
            p1_id=p1.player_id, p2_id=p2.player_id,
            round="SF", match_date=date(2026, 8, 3),
            status="scheduled", source_uid="live-uid",
        )
        ts = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)

        matches.update_live_fields(
            match_id=seeded.match_id, start_ts=ts, status="live",
            match_date_source="atp_scraper",
        )
        after = matches.get(seeded.match_id)
        assert after is not None
        assert after.status == "live"
        assert after.start_ts == ts
        assert after.match_date_source == "atp_scraper"
        assert after.round == "SF"          # untouched
        assert after.p1_id == seeded.p1_id  # untouched

        # Absent match_id → no-op, no row created.
        matches.update_live_fields(
            match_id=424242, start_ts=ts, status="live",
            match_date_source="atp_scraper",
        )
        with session_factory() as s:
            assert s.execute(
                select(m.Match).where(m.Match.match_id == 424242)
            ).scalar_one_or_none() is None

    def test_odds_snapshot_natural_key_dedup(self, session_factory: Any) -> None:
        from sqlalchemy import select

        from tennis.storage.postgres import models as m
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl, OddsSnapshotRepositoryImpl,
            PlayerRepositoryImpl, TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )
        from tennis.storage.postgres.rows import OddsSnapshotRow

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches_repo = MatchRepositoryImpl(session_factory)
        odds = OddsSnapshotRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="OddsCity", country="OD1")
        t = _seed_tournament(tournaments, slug="odds-t", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="oA")
        p2 = _seed_player(players, source_uid="oB")
        match = _seed_match(
            matches_repo, tournament_id=t.tournament_id,
            p1_id=p1.player_id, p2_id=p2.player_id,
            round="QF", match_date=date(2026, 8, 5),
            status="scheduled", source_uid="odds-match",
        )

        captured = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
        # Two inserts with the same natural key (different prices).
        odds.insert(OddsSnapshotRow(
            match_id=match.match_id, bookmaker="pinnacle",
            captured_at=captured,
            p1_decimal=1.90, p2_decimal=1.95,
            p1_implied=0.50, p2_implied=0.50, vig=0.05,
            devig_method="shin",
        ))
        odds.insert(OddsSnapshotRow(
            match_id=match.match_id, bookmaker="pinnacle",
            captured_at=captured,
            p1_decimal=1.85, p2_decimal=2.00,
            p1_implied=0.52, p2_implied=0.48, vig=0.05,
            devig_method="shin",
        ))

        with session_factory() as s:
            rows = s.execute(
                select(m.OddsSnapshot).where(
                    m.OddsSnapshot.match_id == match.match_id,
                    m.OddsSnapshot.bookmaker == "pinnacle",
                    m.OddsSnapshot.captured_at == captured,
                    m.OddsSnapshot.devig_method == "shin",
                )
            ).scalars().all()
            assert len(rows) == 1
            assert float(rows[0].p1_decimal) == 1.85


# ---------------------------------------------------------------------------
# OddsSnapshotRepository.latest_before() — §15.4 inclusive decision boundary
# ---------------------------------------------------------------------------
class TestOddsLatestBeforeBoundary:
    """§15.4 locks the decision-time bound as `captured_at <= as_of_ts`. A snapshot
    captured EXACTLY at the cut must be returned, not dropped (Codex R7 HIGH — the
    real repo previously used strict `<`, diverging from the contract + the fake)."""

    def test_snapshot_at_exact_boundary_is_included(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl, OddsSnapshotRepositoryImpl,
            PlayerRepositoryImpl, TournamentRepositoryImpl, VenueRepositoryImpl,
        )
        from tennis.storage.postgres.rows import OddsSnapshotRow

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches_repo = MatchRepositoryImpl(session_factory)
        odds = OddsSnapshotRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="BoundaryCity", country="BD1")
        t = _seed_tournament(tournaments, slug="boundary-t", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="bndA")
        p2 = _seed_player(players, source_uid="bndB")
        match = _seed_match(
            matches_repo, tournament_id=t.tournament_id,
            p1_id=p1.player_id, p2_id=p2.player_id,
            round="QF", match_date=date(2026, 8, 6),
            status="scheduled", source_uid="boundary-match",
        )

        cut = datetime(2026, 8, 6, 9, 0, tzinfo=UTC)
        odds.insert(OddsSnapshotRow(
            match_id=match.match_id, bookmaker="pinnacle",
            captured_at=cut - timedelta(hours=2),
            p1_decimal=1.90, p2_decimal=1.95,
            p1_implied=0.50, p2_implied=0.50, vig=0.05, devig_method="shin",
        ))
        odds.insert(OddsSnapshotRow(
            match_id=match.match_id, bookmaker="pinnacle", captured_at=cut,
            p1_decimal=1.80, p2_decimal=2.05,
            p1_implied=0.55, p2_implied=0.45, vig=0.05, devig_method="shin",
        ))

        # captured_before == the boundary snapshot's captured_at → it MUST win.
        got = odds.latest_before(
            match_id=match.match_id, bookmaker="pinnacle",
            devig_method="shin", captured_before=cut,
        )
        assert got is not None
        assert got.p1_implied == 0.55  # the at-boundary snapshot, not the earlier one


# ---------------------------------------------------------------------------
# VenueRepository.list_all() — supports the DataAgent OWM venue enumeration (§L5)
# ---------------------------------------------------------------------------
class TestVenueListAll:
    def test_returns_all_venues_ordered_by_id(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import VenueRepositoryImpl
        from tennis.storage.postgres.rows import VenueRow

        venues = VenueRepositoryImpl(session_factory)

        # One venue WITH coordinates, one WITHOUT — mirrors the v1 geocoding gap.
        with_coords = venues.upsert(VenueRow(
            venue_id=compute_venue_id(city="ListAllCoords", country_code="LA1"),
            city="ListAllCoords", country_code="LA1",
            latitude=51.5, longitude=-0.12,
        ))
        without_coords = venues.upsert(VenueRow(
            venue_id=compute_venue_id(city="ListAllNoCoords", country_code="LA2"),
            city="ListAllNoCoords", country_code="LA2",
        ))

        rows = list(venues.list_all())
        ids = [r.venue_id for r in rows]

        # Both seeded venues present, ordered by venue_id (deterministic).
        assert with_coords.venue_id in ids
        assert without_coords.venue_id in ids
        assert ids == sorted(ids)

        # The DataAgent's coord filter (§L5) keeps only lat/lon-bearing venues.
        coord_ids = [
            r.venue_id for r in rows
            if r.latitude is not None and r.longitude is not None
        ]
        assert with_coords.venue_id in coord_ids
        assert without_coords.venue_id not in coord_ids


# ---------------------------------------------------------------------------
# EloSnapshotRepository.career_match_counts() — the prediction-path §M9 source
# ---------------------------------------------------------------------------
class TestEloCareerMatchCounts:
    def test_counts_distinct_matches_per_player(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import (
            EloSnapshotRepositoryImpl,
            MatchRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )
        from tennis.storage.postgres.rows import EloSnapshotRow

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)
        elo = EloSnapshotRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="EloCountCity", country="ELO")
        t = _seed_tournament(tournaments, slug="elo-count", venue_id=v.venue_id)
        p1 = _seed_player(players, source_uid="eloA")
        p2 = _seed_player(players, source_uid="eloB")

        # Two matches between p1 and p2. Each Elo-updating match writes TWO rows
        # per player (overall + surface) — exactly what the EloWalk does — so the
        # DISTINCT match_id count must collapse them to the per-player match count.
        rounds = ["R16", "QF"]
        match_ids = []
        for i, rnd in enumerate(rounds):
            mr = _seed_match(
                matches, tournament_id=t.tournament_id,
                p1_id=p1.player_id, p2_id=p2.player_id,
                round=rnd, match_date=date(2026, 1, 10 + i),
                status="final", source_uid=f"elo-{rnd}",
            )
            match_ids.append(mr.match_id)
            stamp = datetime(2026, 1, 10 + i, 23, 59, tzinfo=UTC)
            for pid in (p1.player_id, p2.player_id):
                for surface in ("overall", "Hard"):
                    elo.insert(EloSnapshotRow(
                        player_id=pid, surface=surface,
                        elo_rating=1500.0, as_of_ts=stamp, match_id=mr.match_id,
                    ))

        counts = elo.career_match_counts()
        # 4 snapshot rows per player (2 matches × overall+surface) → DISTINCT == 2.
        assert counts[p1.player_id] == 2
        assert counts[p2.player_id] == 2

    def test_player_with_no_snapshots_absent_from_map(self, session_factory: Any) -> None:
        from tennis.storage.postgres.impl import (
            EloSnapshotRepositoryImpl,
            PlayerRepositoryImpl,
        )

        # Session-scoped DB is shared across tests, so assert on a specific
        # never-snapshotted player rather than an empty map: a player with no
        # Elo-updating match is simply absent (the EloExtractor defaults to 0).
        players = PlayerRepositoryImpl(session_factory)
        ghost = _seed_player(players, source_uid="eloGhost")
        counts = EloSnapshotRepositoryImpl(session_factory).career_match_counts()
        assert ghost.player_id not in counts


# ---------------------------------------------------------------------------
# MatchStatRepository.list_for_player() — the §M18 bulk serve/return read
# (retires the §M14 per-match get() N+1)
# ---------------------------------------------------------------------------
class TestMatchStatListForPlayer:
    def test_bulk_read_filters_by_player_and_keys_by_match(
        self, session_factory: Any
    ) -> None:
        from tennis.storage.postgres.impl import (
            MatchRepositoryImpl,
            MatchStatRepositoryImpl,
            PlayerRepositoryImpl,
            TournamentRepositoryImpl,
            VenueRepositoryImpl,
        )
        from tennis.storage.postgres.rows import MatchStatRow

        players = PlayerRepositoryImpl(session_factory)
        venues = VenueRepositoryImpl(session_factory)
        tournaments = TournamentRepositoryImpl(session_factory)
        matches = MatchRepositoryImpl(session_factory)
        stats = MatchStatRepositoryImpl(session_factory)

        v = _seed_venue(venues, city="StatBulkCity", country="SBK")
        t = _seed_tournament(tournaments, slug="stat-bulk", venue_id=v.venue_id)
        target = _seed_player(players, source_uid="sbTarget")
        opp = _seed_player(players, source_uid="sbOpp")

        # Three matches target-vs-opp. target has a stat row in m1 and m2 (NOT m3);
        # opp has a stat row in m1 (the decoy that the player filter must exclude).
        match_ids = []
        for i, rnd in enumerate(["R32", "R16", "QF"]):
            mr = _seed_match(
                matches, tournament_id=t.tournament_id,
                p1_id=target.player_id, p2_id=opp.player_id,
                round=rnd, match_date=date(2026, 3, 1 + i),
                status="final", source_uid=f"sb-{rnd}",
            )
            match_ids.append(mr.match_id)
        m1, m2, m3 = match_ids

        stats.upsert(MatchStatRow(
            match_id=m1, player_id=target.player_id, is_winner=True, serve_pts=80,
        ))
        stats.upsert(MatchStatRow(
            match_id=m2, player_id=target.player_id, is_winner=False, serve_pts=70,
        ))
        # Decoy: same match m1, DIFFERENT player — must not appear in target's map.
        stats.upsert(MatchStatRow(
            match_id=m1, player_id=opp.player_id, is_winner=False, serve_pts=99,
        ))

        result = stats.list_for_player(
            player_id=target.player_id, match_ids=[m1, m2, m3]
        )

        # Keyed by match_id; m3 absent (no row); opp's m1 row excluded (player filter).
        assert set(result) == {m1, m2}
        assert all(row.player_id == target.player_id for row in result.values())
        assert result[m1].serve_pts == 80
        assert result[m2].serve_pts == 70

    def test_empty_match_ids_returns_empty_no_query(
        self, session_factory: Any
    ) -> None:
        from tennis.storage.postgres.impl import (
            MatchStatRepositoryImpl,
            PlayerRepositoryImpl,
        )

        # Empty fast-path (contract): returns {} without touching the DB.
        players = PlayerRepositoryImpl(session_factory)
        p = _seed_player(players, source_uid="sbEmpty")
        stats = MatchStatRepositoryImpl(session_factory)
        assert stats.list_for_player(player_id=p.player_id, match_ids=[]) == {}
