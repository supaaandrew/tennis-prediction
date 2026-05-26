"""Concrete Postgres repository implementations.

Each class here satisfies the corresponding Protocol from
`repositories.py`. Constructors take a single `session_factory` callable
that returns a context manager yielding a SQLAlchemy `Session` — this
matches how `PostgresSessionFactory.session` works and how the DI
container in `core/di.py` registers factories.

Locked decisions baked into the implementations:
  - Every upsert uses `INSERT ... ON CONFLICT DO UPDATE` on the natural
    unique key from the migrations. Never DELETE + INSERT.
  - `MatchRepositoryImpl.for_training` / `for_prediction` enforce status
    AND tier in SQL — never in Python — via a JOIN to `tournaments`.
    Status set is fixed per C4; tier list defaults to C3's
    `(GS, Masters1000, ATP500, ATP250)`.
  - `WeatherObservationRepositoryImpl.upsert` writes the prior row to
    `weather_revisions` before overwriting (audit trail, A12).
  - `EloSnapshotRepositoryImpl` is INSERT-only; PK violation raises
    `IdempotencyError`.
  - `DeadLetterRepositoryImpl.append` swallows ALL exceptions; the
    ingest path is already failing when it lands here and a second
    exception would crash the run. Logged via structlog instead.
  - Every method that takes a datetime checks tz-awareness up front.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from datetime import date, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from tennis.core.errors import IdempotencyError, StorageError
from tennis.core.logging import get_logger
from tennis.storage.postgres import models as m
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    DevigMethod,
    EloSnapshotRow,
    EloSurface,
    FeatureMatrixRow,
    FeatureSpecRow,
    IngestWatermarkRow,
    MatchRow,
    MatchStatRow,
    ModelRegistryRow,
    OddsSnapshotRow,
    PipelineRunRow,
    PlayerAliasRow,
    PlayerRankingRow,
    PlayerRow,
    PredictionRow,
    RunStatusLit,
    TournamentRow,
    VenueRow,
    WeatherObservationRow,
    WeatherRevisionRow,
)

_logger = get_logger("tennis.storage.postgres.impl")

SessionCallable = Callable[[], AbstractContextManager[Session]]

DEFAULT_INCLUDED_TIERS: tuple[str, ...] = ("GS", "Masters1000", "ATP500", "ATP250")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_tz_aware(value: datetime, name: str) -> None:
    """Reject naive datetimes; the project is UTC-only by contract."""
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC)")


def _orm_to_row(orm: Any, row_cls: type) -> Any:
    """Generic ORM → Row converter using dataclass field names.

    ORM column names match Row field names by design (rows.py mirrors
    the migrations). For JSONB-array columns the ORM returns a list; we
    coerce to tuple for the single Row field that requires it
    (`PlayerRow.aliases`).
    """
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(row_cls):
        if not hasattr(orm, f.name):
            continue
        v = getattr(orm, f.name)
        if isinstance(v, list) and f.name == "aliases":
            v = tuple(v)
        kwargs[f.name] = v
    return row_cls(**kwargs)


def _row_to_dict(row: Any, *, drop_none_for: tuple[str, ...] = ()) -> dict[str, Any]:
    """Generic Row → dict for SQLAlchemy INSERT statements."""
    d: dict[str, Any] = {}
    for f in dataclasses.fields(row):
        v = getattr(row, f.name)
        if v is None and f.name in drop_none_for:
            continue
        if isinstance(v, tuple) and f.name == "aliases":
            v = list(v)
        d[f.name] = v
    return d


_SERVER_MANAGED = ("created_at", "updated_at")

# Postgres SQLSTATE for unique_violation. Used to distinguish duplicate-PK
# violations (legitimate idempotency) from other integrity failures
# (FK violations, CHECK failures, etc.) — those must NOT be converted to
# IdempotencyError or adapters will silently retry on real corruption.
_PG_UNIQUE_VIOLATION = "23505"


def _is_unique_violation_on(exc: IntegrityError, constraint_name: str) -> bool:
    """True iff `exc` is a Postgres unique_violation (SQLSTATE 23505) on
    the specified constraint.

    Detection order:
      1. Inspect the underlying DBAPI error's `pgcode` and
         `diag.constraint_name` (psycopg3 path).
      2. If neither is available (e.g. unit-test mocks), fall back to a
         substring scan of the formatted exception message.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return constraint_name in str(exc)
    pgcode = getattr(orig, "pgcode", None)
    if isinstance(pgcode, str) and pgcode != _PG_UNIQUE_VIOLATION:
        return False
    diag = getattr(orig, "diag", None)
    if diag is not None:
        cname = getattr(diag, "constraint_name", None)
        if isinstance(cname, str):
            return cname == constraint_name
    return constraint_name in str(exc)


# ===========================================================================
# Players + rankings + venues + tournaments (migration 001)
# ===========================================================================
class PlayerRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, player_id: int) -> PlayerRow | None:
        with self._sf() as s:
            o = s.get(m.Player, player_id)
            return _orm_to_row(o, PlayerRow) if o else None

    def get_by_source(self, *, source: str, source_uid: str) -> PlayerRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.Player).where(
                    m.Player.source == source,
                    m.Player.source_uid == source_uid,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, PlayerRow) if o else None

    def upsert(self, row: PlayerRow) -> PlayerRow:
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.Player).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["source", "source_uid"],
                set_={
                    "full_name": stmt.excluded.full_name,
                    "country_code": stmt.excluded.country_code,
                    "date_of_birth": stmt.excluded.date_of_birth,
                    "dominant_hand": stmt.excluded.dominant_hand,
                    "backhand": stmt.excluded.backhand,
                    "height_cm": stmt.excluded.height_cm,
                    "pro_since": stmt.excluded.pro_since,
                    "sackmann_atp_id": stmt.excluded.sackmann_atp_id,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.Player).where(
                    m.Player.source == row.source,
                    m.Player.source_uid == row.source_uid,
                )
            ).scalar_one()
            return _orm_to_row(o, PlayerRow)


class PlayerRankingRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, *, player_id: int, ranking_date: date) -> PlayerRankingRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.PlayerRanking).where(
                    m.PlayerRanking.player_id == player_id,
                    m.PlayerRanking.ranking_date == ranking_date,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, PlayerRankingRow) if o else None

    def upsert(self, row: PlayerRankingRow) -> PlayerRankingRow:
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.PlayerRanking).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["player_id", "ranking_date"],
                set_={"rank": stmt.excluded.rank, "points": stmt.excluded.points},
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.PlayerRanking).where(
                    m.PlayerRanking.player_id == row.player_id,
                    m.PlayerRanking.ranking_date == row.ranking_date,
                )
            ).scalar_one()
            return _orm_to_row(o, PlayerRankingRow)

    def latest_before(
        self, *, player_id: int, on_or_before: date
    ) -> PlayerRankingRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.PlayerRanking)
                .where(
                    m.PlayerRanking.player_id == player_id,
                    m.PlayerRanking.ranking_date <= on_or_before,
                )
                .order_by(m.PlayerRanking.ranking_date.desc())
                .limit(1)
            ).scalar_one_or_none()
            return _orm_to_row(o, PlayerRankingRow) if o else None


class VenueRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, venue_id: int) -> VenueRow | None:
        with self._sf() as s:
            o = s.get(m.Venue, venue_id)
            return _orm_to_row(o, VenueRow) if o else None

    def get_by_city_country(
        self, *, city: str, country_code: str
    ) -> VenueRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.Venue).where(
                    m.Venue.city == city,
                    m.Venue.country_code == country_code,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, VenueRow) if o else None

    def list_all(self) -> Sequence[VenueRow]:
        with self._sf() as s:
            rows = (
                s.execute(select(m.Venue).order_by(m.Venue.venue_id))
                .scalars()
                .all()
            )
            return [_orm_to_row(o, VenueRow) for o in rows]

    def upsert(self, row: VenueRow) -> VenueRow:
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.Venue).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["city", "country_code"],
                set_={
                    "latitude": stmt.excluded.latitude,
                    "longitude": stmt.excluded.longitude,
                    "altitude_m": stmt.excluded.altitude_m,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.Venue).where(
                    m.Venue.city == row.city,
                    m.Venue.country_code == row.country_code,
                )
            ).scalar_one()
            return _orm_to_row(o, VenueRow)


class TournamentRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, tournament_id: int) -> TournamentRow | None:
        with self._sf() as s:
            o = s.get(m.Tournament, tournament_id)
            return _orm_to_row(o, TournamentRow) if o else None

    def get_by_season_slug(
        self, *, season: int, slug: str
    ) -> TournamentRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.Tournament).where(
                    m.Tournament.season == season,
                    m.Tournament.slug == slug,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, TournamentRow) if o else None

    def upsert(self, row: TournamentRow) -> TournamentRow:
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.Tournament).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["season", "slug"],
                set_={
                    "name": stmt.excluded.name,
                    "tier": stmt.excluded.tier,
                    "surface": stmt.excluded.surface,
                    "indoor": stmt.excluded.indoor,
                    "draw_size": stmt.excluded.draw_size,
                    "venue_id": stmt.excluded.venue_id,
                    "start_date": stmt.excluded.start_date,
                    "end_date": stmt.excluded.end_date,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.Tournament).where(
                    m.Tournament.season == row.season,
                    m.Tournament.slug == row.slug,
                )
            ).scalar_one()
            return _orm_to_row(o, TournamentRow)


# ===========================================================================
# Matches + match stats + odds snapshots (migrations 002, 012)
# ===========================================================================
class MatchRepositoryImpl:
    """Match persistence + policy-locked filters.

    `for_training` enforces `status='final'` AND tier ∈ `included_tiers`
    in SQL. `for_prediction` enforces `status IN ('scheduled','live')`
    AND tier ∈ `included_tiers`. The tier list defaults to C3 values;
    the DI container can inject the configured value if it ever drifts.
    """

    def __init__(
        self,
        session_factory: SessionCallable,
        *,
        included_tiers: tuple[str, ...] = DEFAULT_INCLUDED_TIERS,
    ) -> None:
        self._sf = session_factory
        self._included_tiers = tuple(included_tiers)

    def get(self, match_id: int) -> MatchRow | None:
        with self._sf() as s:
            o = s.get(m.Match, match_id)
            return _orm_to_row(o, MatchRow) if o else None

    def get_by_source(self, *, source: str, source_uid: str) -> MatchRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.Match).where(
                    m.Match.source == source,
                    m.Match.source_uid == source_uid,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, MatchRow) if o else None

    def find_by_players_and_date(
        self,
        *,
        player_a_id: int,
        player_b_id: int,
        match_date: date,
        window_days: int = 1,
    ) -> MatchRow | None:
        """See `MatchRepository.find_by_players_and_date` (J1). Returns the
        single in-range match, or None when zero OR >1 candidates exist."""
        lo = match_date - timedelta(days=window_days)
        hi = match_date + timedelta(days=window_days)
        with self._sf() as s:
            stmt = (
                select(m.Match)
                .where(m.Match.match_date >= lo)
                .where(m.Match.match_date <= hi)
                .where(
                    or_(
                        and_(
                            m.Match.p1_id == player_a_id,
                            m.Match.p2_id == player_b_id,
                        ),
                        and_(
                            m.Match.p1_id == player_b_id,
                            m.Match.p2_id == player_a_id,
                        ),
                    )
                )
                # Fetch two so we can detect ambiguity without scanning all.
                .limit(2)
            )
            rows = s.execute(stmt).scalars().all()
            if len(rows) != 1:
                return None
            return _orm_to_row(rows[0], MatchRow)

    def upsert(self, row: MatchRow) -> MatchRow:
        """Insert or merge, reconciling on the `match_id` PRIMARY KEY (K4).

        Dedup is keyed on `match_id`, not `(source, source_uid)`: the scraper
        and Sackmann describe the same logical match with the same `match_id`
        (C1) but different `source_uid` (K2). Keying the conflict on the PK
        lets either source write second without an `IntegrityError`. `start_ts`
        is preserved via COALESCE so a NULL (Sackmann) never clobbers a known
        scraper timestamp (H3/§15.2).

        Identity fields are NEVER rewritten on conflict: `source`, `source_uid`,
        `match_date`, `tournament_id`, `round`, `p1_id`, `p2_id`. The row keeps
        the originating source of whoever inserted it first. Rewriting
        `source`/`source_uid` here would (a) risk a unique violation on the
        secondary `(source, source_uid)` index that `ON CONFLICT (match_id)`
        does not catch, and (b) churn identity on every cross-source merge
        (Codex HIGH). Only mutable match facts merge in (result, status, dates).
        """
        if row.start_ts is not None:
            _require_tz_aware(row.start_ts, "start_ts")
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.Match).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_id"],
                set_={
                    # COALESCE: keep an existing start_ts when the incoming
                    # row carries NULL (scraper owns start_ts; Sackmann NULLs it).
                    "start_ts": func.coalesce(
                        stmt.excluded.start_ts, m.Match.start_ts
                    ),
                    # Mutable match facts — Sackmann finalizing a scraper-created
                    # row MUST be able to fill these in (else a final row would
                    # carry no winner/score). Result fields are not identity.
                    "winner_id": stmt.excluded.winner_id,
                    "loser_id": stmt.excluded.loser_id,
                    "score": stmt.excluded.score,
                    "best_of": stmt.excluded.best_of,
                    "sets_played": stmt.excluded.sets_played,
                    "minutes": stmt.excluded.minutes,
                    "retired": stmt.excluded.retired,
                    "walkover": stmt.excluded.walkover,
                    "status": stmt.excluded.status,
                    "match_date_source": stmt.excluded.match_date_source,
                    # NOT updated (identity / hash inputs, kept from first writer;
                    # all are necessarily identical on a match_id conflict):
                    #   source, source_uid, match_date, tournament_id, round,
                    #   p1_id, p2_id.
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.get(m.Match, row.match_id)
            return _orm_to_row(o, MatchRow)

    def update_live_fields(
        self,
        *,
        match_id: int,
        start_ts: datetime | None,
        status: str,
        match_date_source: str,
    ) -> None:
        """Update only the scraper-owned fields on an existing row (K1).

        No-op when `match_id` doesn't exist — logs a warning and returns
        rather than raising, so the scraper's reconciliation path can fall
        through to a full upsert without special-casing the race.
        """
        if start_ts is not None:
            _require_tz_aware(start_ts, "start_ts")
        with self._sf() as s:
            o = s.get(m.Match, match_id)
            if o is None:
                _logger.warning("match_update_live_fields_missing", match_id=match_id)
                return
            o.start_ts = start_ts
            o.status = status
            o.match_date_source = match_date_source
            s.flush()

    def for_training(
        self, *, season_start: int, season_end: int
    ) -> Iterable[MatchRow]:
        """Status='final' AND tier ∈ included_tiers — both filtered in SQL."""
        with self._sf() as s:
            stmt = (
                select(m.Match)
                .join(m.Tournament, m.Match.tournament_id == m.Tournament.tournament_id)
                .where(m.Match.status == "final")
                .where(m.Tournament.tier.in_(self._included_tiers))
                .where(m.Tournament.season >= season_start)
                .where(m.Tournament.season <= season_end)
                .order_by(m.Match.match_date.asc(), m.Match.match_id.asc())
            )
            return [_orm_to_row(o, MatchRow) for o in s.execute(stmt).scalars().all()]

    def for_prediction(
        self, *, as_of: datetime, lookforward_days: int
    ) -> Iterable[MatchRow]:
        """Status IN ('scheduled','live') AND tier ∈ included_tiers."""
        _require_tz_aware(as_of, "as_of")
        upper = as_of + timedelta(days=lookforward_days)
        with self._sf() as s:
            # Window is (as_of, as_of + N days]; matches with start_ts use it,
            # matches without start_ts fall back to match_date at UTC midnight.
            stmt = (
                select(m.Match)
                .join(m.Tournament, m.Match.tournament_id == m.Tournament.tournament_id)
                .where(m.Match.status.in_(("scheduled", "live")))
                .where(m.Tournament.tier.in_(self._included_tiers))
                .where(
                    func.coalesce(
                        m.Match.start_ts,
                        func.cast(m.Match.match_date, m.Match.start_ts.type),
                    )
                    > as_of
                )
                .where(
                    func.coalesce(
                        m.Match.start_ts,
                        func.cast(m.Match.match_date, m.Match.start_ts.type),
                    )
                    <= upper
                )
                .order_by(
                    func.coalesce(m.Match.start_ts, m.Match.match_date).asc(),
                    m.Match.match_id.asc(),
                )
            )
            return [_orm_to_row(o, MatchRow) for o in s.execute(stmt).scalars().all()]

    def mark_intraday_conflict(self, *, match_id: int, conflict: bool) -> None:
        with self._sf() as s:
            o = s.get(m.Match, match_id)
            if o is None:
                return
            o.intraday_conflict = conflict


class MatchStatRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, *, match_id: int, player_id: int) -> MatchStatRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.MatchStat).where(
                    m.MatchStat.match_id == match_id,
                    m.MatchStat.player_id == player_id,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, MatchStatRow) if o else None

    def list_for_match(self, match_id: int) -> Sequence[MatchStatRow]:
        with self._sf() as s:
            rows = s.execute(
                select(m.MatchStat).where(m.MatchStat.match_id == match_id)
            ).scalars().all()
            return [_orm_to_row(o, MatchStatRow) for o in rows]

    def list_for_player(
        self, *, player_id: int, match_ids: Sequence[int]
    ) -> Mapping[int, MatchStatRow]:
        """All of `player_id`'s match_stats among `match_ids`, keyed by match_id.

        The bulk dual of `get` (R6b, §M18): the serve/return extractor fetches a
        player's whole prior-match stat set in ONE query instead of N per-match
        `get` calls (retires the §M14 N+1). `(match_id, player_id)` is the PK and we
        filter on a single `player_id`, so each match_id maps to exactly one row.
        """
        # Empty fast-path: no DB round-trip on empty input (contract).
        if not match_ids:
            return {}
        # No IN-list chunking (v1): a player's full career (~2000 ids) is well within
        # Postgres IN-list limits; documented accepted limitation (§M18).
        # A DB/IO failure is raised as a TYPED StorageError (not a raw SQLAlchemy
        # error) so the serve/return extractor's degrade path can catch exactly the
        # intended failure class and let genuine bugs propagate loudly (§M18).
        try:
            with self._sf() as s:
                rows = s.execute(
                    select(m.MatchStat).where(
                        m.MatchStat.player_id == player_id,
                        m.MatchStat.match_id.in_(tuple(match_ids)),
                    )
                ).scalars().all()
                return {o.match_id: _orm_to_row(o, MatchStatRow) for o in rows}
        except SQLAlchemyError as exc:
            raise StorageError("MatchStat.list_for_player query failed") from exc

    def upsert(self, row: MatchStatRow) -> MatchStatRow:
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.MatchStat).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_id", "player_id"],
                set_={
                    k: getattr(stmt.excluded, k)
                    for k in (
                        "is_winner", "aces", "double_faults", "serve_pts",
                        "first_in", "first_won", "second_won", "serve_games",
                        "bp_saved", "bp_faced", "winners", "unforced_errors",
                        "net_points_won", "net_points_total",
                    )
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.MatchStat).where(
                    m.MatchStat.match_id == row.match_id,
                    m.MatchStat.player_id == row.player_id,
                )
            ).scalar_one()
            return _orm_to_row(o, MatchStatRow)


class OddsSnapshotRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def insert(self, row: OddsSnapshotRow) -> OddsSnapshotRow:
        _require_tz_aware(row.captured_at, "captured_at")
        values = _row_to_dict(row, drop_none_for=("snapshot_id", "created_at"))
        with self._sf() as s:
            stmt = pg_insert(m.OddsSnapshot).values(**values)
            # Natural unique key for dedup; conflict means "we already have it".
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    "match_id", "bookmaker", "market", "captured_at", "devig_method"
                ],
                set_={
                    "is_opening": stmt.excluded.is_opening,
                    "is_closing": stmt.excluded.is_closing,
                    "p1_decimal": stmt.excluded.p1_decimal,
                    "p2_decimal": stmt.excluded.p2_decimal,
                    "p1_implied": stmt.excluded.p1_implied,
                    "p2_implied": stmt.excluded.p2_implied,
                    "vig": stmt.excluded.vig,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.OddsSnapshot).where(
                    m.OddsSnapshot.match_id == row.match_id,
                    m.OddsSnapshot.bookmaker == row.bookmaker,
                    m.OddsSnapshot.market == row.market,
                    m.OddsSnapshot.captured_at == row.captured_at,
                    m.OddsSnapshot.devig_method == row.devig_method,
                )
            ).scalar_one()
            return _orm_to_row(o, OddsSnapshotRow)

    def list_for_match(
        self, *, match_id: int, bookmaker: str | None = None
    ) -> Sequence[OddsSnapshotRow]:
        with self._sf() as s:
            stmt = select(m.OddsSnapshot).where(m.OddsSnapshot.match_id == match_id)
            if bookmaker is not None:
                stmt = stmt.where(m.OddsSnapshot.bookmaker == bookmaker)
            stmt = stmt.order_by(m.OddsSnapshot.captured_at.asc())
            return [_orm_to_row(o, OddsSnapshotRow) for o in s.execute(stmt).scalars().all()]

    def opening(
        self, *, match_id: int, bookmaker: str, devig_method: DevigMethod
    ) -> OddsSnapshotRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.OddsSnapshot).where(
                    m.OddsSnapshot.match_id == match_id,
                    m.OddsSnapshot.bookmaker == bookmaker,
                    m.OddsSnapshot.devig_method == devig_method,
                    m.OddsSnapshot.is_opening.is_(True),
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, OddsSnapshotRow) if o else None

    def closing(
        self, *, match_id: int, bookmaker: str, devig_method: DevigMethod
    ) -> OddsSnapshotRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.OddsSnapshot).where(
                    m.OddsSnapshot.match_id == match_id,
                    m.OddsSnapshot.bookmaker == bookmaker,
                    m.OddsSnapshot.devig_method == devig_method,
                    m.OddsSnapshot.is_closing.is_(True),
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, OddsSnapshotRow) if o else None

    def latest_before(
        self,
        *,
        match_id: int,
        bookmaker: str,
        devig_method: DevigMethod,
        captured_before: datetime,
    ) -> OddsSnapshotRow | None:
        _require_tz_aware(captured_before, "captured_before")
        with self._sf() as s:
            o = s.execute(
                select(m.OddsSnapshot)
                .where(
                    m.OddsSnapshot.match_id == match_id,
                    m.OddsSnapshot.bookmaker == bookmaker,
                    m.OddsSnapshot.devig_method == devig_method,
                    m.OddsSnapshot.captured_at < captured_before,
                )
                .order_by(m.OddsSnapshot.captured_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            return _orm_to_row(o, OddsSnapshotRow) if o else None


# ===========================================================================
# Weather (migration 003) — revision-audit on overwrite
# ===========================================================================
class WeatherObservationRepositoryImpl:
    """Per A12: any UPDATE writes the prior row to `weather_revisions`
    first so silent provider drift is auditable. INSERT-of-new-row
    skips the audit.

    Race-safety: a previous implementation did a read-then-write in
    Python — two concurrent writers could both observe "no existing
    row", and the loser would still hit the conflict path and silently
    overwrite without producing a revision. The fix has two parts:

      1. A transaction-scoped advisory lock keyed on the natural
         `(venue_id, observed_at, source)` tuple serializes writers to
         the same PK so the second writer always sees the first
         writer's committed row.
      2. A single SQL statement that captures the prior row (with
         `FOR UPDATE`), conditionally inserts a `weather_revisions`
         row in the same statement, and then performs the upsert.
         The revision insert and the row update are atomic — either
         both commit or both abort.
    """

    # Column names persisted by the upsert (excludes PK columns and
    # server-managed timestamps which are populated by defaults).
    _UPDATABLE_COLS: tuple[str, ...] = (
        "is_forecast", "temp_c", "humidity_pct", "wind_speed_ms",
        "wind_dir_deg", "pressure_hpa", "precip_mm", "cloud_pct",
        "forecast_horizon_h",
    )

    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(
        self, *, venue_id: int, observed_at: datetime, source: str
    ) -> WeatherObservationRow | None:
        _require_tz_aware(observed_at, "observed_at")
        with self._sf() as s:
            o = s.execute(
                select(m.WeatherObservation).where(
                    m.WeatherObservation.venue_id == venue_id,
                    m.WeatherObservation.observed_at == observed_at,
                    m.WeatherObservation.source == source,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, WeatherObservationRow) if o else None

    def upsert(self, row: WeatherObservationRow) -> WeatherObservationRow:
        _require_tz_aware(row.observed_at, "observed_at")
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        new_payload_json = json.dumps(_jsonable(values))
        lock_key = (
            f"weather:{row.venue_id}:{row.observed_at.isoformat()}:{row.source}"
        )

        # The revision-and-upsert statement. Data-modifying CTEs run
        # atomically with the primary INSERT under the same snapshot —
        # the revision either lands or the whole statement aborts.
        sql = text(
            """
            WITH old AS (
                SELECT * FROM weather_observations
                WHERE venue_id = :venue_id
                  AND observed_at = :observed_at
                  AND source = :source
                FOR UPDATE
            ),
            revision AS (
                INSERT INTO weather_revisions
                    (venue_id, observed_at, source, previous_row, new_row)
                SELECT old.venue_id, old.observed_at, old.source,
                       row_to_json(old.*)::jsonb,
                       CAST(:new_payload AS jsonb)
                FROM old
                WHERE old.venue_id IS NOT NULL
            )
            INSERT INTO weather_observations
                (venue_id, observed_at, source, is_forecast, temp_c,
                 humidity_pct, wind_speed_ms, wind_dir_deg, pressure_hpa,
                 precip_mm, cloud_pct, forecast_horizon_h)
            VALUES
                (:venue_id, :observed_at, :source, :is_forecast, :temp_c,
                 :humidity_pct, :wind_speed_ms, :wind_dir_deg, :pressure_hpa,
                 :precip_mm, :cloud_pct, :forecast_horizon_h)
            ON CONFLICT (venue_id, observed_at, source) DO UPDATE SET
                is_forecast        = EXCLUDED.is_forecast,
                temp_c             = EXCLUDED.temp_c,
                humidity_pct       = EXCLUDED.humidity_pct,
                wind_speed_ms      = EXCLUDED.wind_speed_ms,
                wind_dir_deg       = EXCLUDED.wind_dir_deg,
                pressure_hpa       = EXCLUDED.pressure_hpa,
                precip_mm          = EXCLUDED.precip_mm,
                cloud_pct          = EXCLUDED.cloud_pct,
                forecast_horizon_h = EXCLUDED.forecast_horizon_h
            """
        )

        with self._sf() as s:
            # Advisory lock — serializes concurrent writers on this PK so
            # the FOR UPDATE in the CTE actually observes prior commits.
            # `hashtext(key)` is int4; cast to bigint for the 1-arg lock
            # form. Released automatically on COMMIT/ROLLBACK.
            s.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:k)::bigint)"),
                {"k": lock_key},
            )
            s.execute(
                sql,
                {
                    "venue_id": row.venue_id,
                    "observed_at": row.observed_at,
                    "source": row.source,
                    "is_forecast": row.is_forecast,
                    "temp_c": row.temp_c,
                    "humidity_pct": row.humidity_pct,
                    "wind_speed_ms": row.wind_speed_ms,
                    "wind_dir_deg": row.wind_dir_deg,
                    "pressure_hpa": row.pressure_hpa,
                    "precip_mm": row.precip_mm,
                    "cloud_pct": row.cloud_pct,
                    "forecast_horizon_h": row.forecast_horizon_h,
                    "new_payload": new_payload_json,
                },
            )
            o = s.execute(
                select(m.WeatherObservation).where(
                    m.WeatherObservation.venue_id == row.venue_id,
                    m.WeatherObservation.observed_at == row.observed_at,
                    m.WeatherObservation.source == row.source,
                )
            ).scalar_one()
            return _orm_to_row(o, WeatherObservationRow)

    def nearest_at_or_before(
        self,
        *,
        venue_id: int,
        target_ts: datetime,
        source: str,
        max_age_hours: int,
    ) -> WeatherObservationRow | None:
        _require_tz_aware(target_ts, "target_ts")
        oldest_ok = target_ts - timedelta(hours=max_age_hours)
        with self._sf() as s:
            o = s.execute(
                select(m.WeatherObservation)
                .where(
                    m.WeatherObservation.venue_id == venue_id,
                    m.WeatherObservation.source == source,
                    m.WeatherObservation.observed_at <= target_ts,
                    m.WeatherObservation.observed_at >= oldest_ok,
                )
                .order_by(m.WeatherObservation.observed_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            return _orm_to_row(o, WeatherObservationRow) if o else None


class WeatherRevisionRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def append(self, row: WeatherRevisionRow) -> WeatherRevisionRow:
        _require_tz_aware(row.observed_at, "observed_at")
        values = _row_to_dict(row, drop_none_for=("revision_id", "revised_at"))
        with self._sf() as s:
            orm = m.WeatherRevision(**values)
            s.add(orm)
            s.flush()
            return _orm_to_row(orm, WeatherRevisionRow)


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Strip values JSONB can't take. datetimes/dates → ISO strings."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ===========================================================================
# Features (migration 004)
# ===========================================================================
class FeatureSpecRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, *, feature_key: str, version: int) -> FeatureSpecRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.FeatureSpec).where(
                    m.FeatureSpec.feature_key == feature_key,
                    m.FeatureSpec.version == version,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, FeatureSpecRow) if o else None

    def list_active(self, *, feature_set: str) -> Sequence[FeatureSpecRow]:
        # feature_set is not stored on feature_specs; the convention is
        # that every spec is active for every feature_set unless the
        # caller filters by feature_key prefix. v1 returns all specs.
        del feature_set  # placeholder for future per-set scoping
        with self._sf() as s:
            rows = s.execute(select(m.FeatureSpec)).scalars().all()
            return [_orm_to_row(o, FeatureSpecRow) for o in rows]

    def upsert(self, row: FeatureSpecRow) -> FeatureSpecRow:
        values = _row_to_dict(row, drop_none_for=("introduced_at",))
        with self._sf() as s:
            stmt = pg_insert(m.FeatureSpec).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["feature_key", "version"],
                set_={
                    "dtype": stmt.excluded.dtype,
                    "description": stmt.excluded.description,
                    "formula_ref": stmt.excluded.formula_ref,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.FeatureSpec).where(
                    m.FeatureSpec.feature_key == row.feature_key,
                    m.FeatureSpec.version == row.version,
                )
            ).scalar_one()
            return _orm_to_row(o, FeatureSpecRow)


class FeatureMatrixRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(
        self, *, match_id: int, feature_set: str
    ) -> FeatureMatrixRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.FeatureMatrix).where(
                    m.FeatureMatrix.match_id == match_id,
                    m.FeatureMatrix.feature_set == feature_set,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, FeatureMatrixRow) if o else None

    def upsert(self, row: FeatureMatrixRow) -> FeatureMatrixRow:
        _require_tz_aware(row.as_of_ts, "as_of_ts")
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        # The DB trigger `fm_no_lookahead` enforces PIT and will raise
        # on a leaky row; we do NOT catch that here — let it propagate.
        with self._sf() as s:
            stmt = pg_insert(m.FeatureMatrix).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_id", "feature_set"],
                set_={
                    "perspective": stmt.excluded.perspective,
                    "as_of_ts": stmt.excluded.as_of_ts,
                    "payload": stmt.excluded.payload,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.FeatureMatrix).where(
                    m.FeatureMatrix.match_id == row.match_id,
                    m.FeatureMatrix.feature_set == row.feature_set,
                )
            ).scalar_one()
            return _orm_to_row(o, FeatureMatrixRow)

    def list_for_matches(
        self, *, match_ids: Sequence[int], feature_set: str
    ) -> Sequence[FeatureMatrixRow]:
        if not match_ids:
            return []
        with self._sf() as s:
            rows = s.execute(
                select(m.FeatureMatrix).where(
                    m.FeatureMatrix.match_id.in_(tuple(match_ids)),
                    m.FeatureMatrix.feature_set == feature_set,
                )
            ).scalars().all()
            return [_orm_to_row(o, FeatureMatrixRow) for o in rows]


# ===========================================================================
# Predictions + model registry (migration 005)
# ===========================================================================
class ModelRegistryRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, version: str) -> ModelRegistryRow | None:
        with self._sf() as s:
            o = s.get(m.ModelRegistry, version)
            return _orm_to_row(o, ModelRegistryRow) if o else None

    def active(self) -> ModelRegistryRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.ModelRegistry).where(m.ModelRegistry.is_active.is_(True))
            ).scalar_one_or_none()
            return _orm_to_row(o, ModelRegistryRow) if o else None

    def insert(self, row: ModelRegistryRow) -> ModelRegistryRow:
        _require_tz_aware(row.trained_at, "trained_at")
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            orm = m.ModelRegistry(**values)
            s.add(orm)
            s.flush()
            return _orm_to_row(orm, ModelRegistryRow)

    def activate(self, version: str) -> None:
        with self._sf() as s:
            # Atomicity: deactivate the current active row (if any) THEN
            # activate the target row, in a single transaction.
            s.execute(
                m.ModelRegistry.__table__.update()
                .where(m.ModelRegistry.is_active.is_(True))
                .values(is_active=False)
            )
            s.execute(
                m.ModelRegistry.__table__.update()
                .where(m.ModelRegistry.version == version)
                .values(is_active=True)
            )


class PredictionRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(
        self, *, match_id: int, model_version: str
    ) -> PredictionRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.Prediction).where(
                    m.Prediction.match_id == match_id,
                    m.Prediction.model_version == model_version,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, PredictionRow) if o else None

    def upsert(self, row: PredictionRow) -> PredictionRow:
        _require_tz_aware(row.predicted_at, "predicted_at")
        values = _row_to_dict(row, drop_none_for=("prediction_id", "created_at"))
        with self._sf() as s:
            stmt = pg_insert(m.Prediction).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_id", "model_version"],
                set_={
                    k: getattr(stmt.excluded, k)
                    for k in (
                        "predicted_at", "p1_prob_raw", "p1_prob_cal",
                        "p1_implied_open", "p1_implied_close",
                        "p1_implied_decision",
                        "edge_p1_shin", "edge_p2_shin",
                        "edge_p1_proportional", "edge_p2_proportional",
                        "kelly_fraction_p1", "kelly_fraction_p2",
                        "odds_drift_to_close",
                    )
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.Prediction).where(
                    m.Prediction.match_id == row.match_id,
                    m.Prediction.model_version == row.model_version,
                )
            ).scalar_one()
            return _orm_to_row(o, PredictionRow)

    def list_for_window(
        self, *, model_version: str, since: datetime, until: datetime
    ) -> Sequence[PredictionRow]:
        _require_tz_aware(since, "since")
        _require_tz_aware(until, "until")
        with self._sf() as s:
            rows = s.execute(
                select(m.Prediction)
                .where(
                    m.Prediction.model_version == model_version,
                    m.Prediction.predicted_at >= since,
                    m.Prediction.predicted_at < until,
                )
                .order_by(m.Prediction.predicted_at.asc())
            ).scalars().all()
            return [_orm_to_row(o, PredictionRow) for o in rows]


# ===========================================================================
# Pipeline lineage (migrations 006, 009)
# ===========================================================================
class PipelineRunRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(
        self, *, run_id: UUID, agent: str, attempt: int = 1
    ) -> PipelineRunRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.PipelineRun).where(
                    m.PipelineRun.run_id == run_id,
                    m.PipelineRun.agent == agent,
                    m.PipelineRun.attempt == attempt,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, PipelineRunRow) if o else None

    def insert(self, row: PipelineRunRow) -> PipelineRunRow:
        _require_tz_aware(row.started_at, "started_at")
        if row.finished_at is not None:
            _require_tz_aware(row.finished_at, "finished_at")
        if row.last_heartbeat_at is not None:
            _require_tz_aware(row.last_heartbeat_at, "last_heartbeat_at")
        values = _row_to_dict(row)
        # metrics is a dict default {} on the row; pass through as JSONB.
        with self._sf() as s:
            orm = m.PipelineRun(**values)
            s.add(orm)
            s.flush()
            return _orm_to_row(orm, PipelineRunRow)

    def update_status(
        self,
        *,
        run_id: UUID,
        agent: str,
        attempt: int,
        status: RunStatusLit,
        finished_at: datetime | None = None,
        metrics: dict | None = None,
        error: dict | None = None,
    ) -> None:
        if finished_at is not None:
            _require_tz_aware(finished_at, "finished_at")
        update_values: dict[str, Any] = {"status": status}
        if finished_at is not None:
            update_values["finished_at"] = finished_at
        if metrics is not None:
            update_values["metrics"] = metrics
        if error is not None:
            update_values["error"] = error
        with self._sf() as s:
            s.execute(
                m.PipelineRun.__table__.update()
                .where(
                    m.PipelineRun.run_id == run_id,
                    m.PipelineRun.agent == agent,
                    m.PipelineRun.attempt == attempt,
                )
                .values(**update_values)
            )

    def heartbeat(
        self, *, run_id: UUID, agent: str, attempt: int, now: datetime
    ) -> None:
        _require_tz_aware(now, "now")
        with self._sf() as s:
            s.execute(
                m.PipelineRun.__table__.update()
                .where(
                    m.PipelineRun.run_id == run_id,
                    m.PipelineRun.agent == agent,
                    m.PipelineRun.attempt == attempt,
                )
                .values(last_heartbeat_at=now)
            )

    def orphans(
        self, *, orphan_after_s: int, now: datetime
    ) -> Sequence[PipelineRunRow]:
        _require_tz_aware(now, "now")
        threshold = now - timedelta(seconds=orphan_after_s)
        with self._sf() as s:
            rows = s.execute(
                select(m.PipelineRun).where(
                    m.PipelineRun.status == "running",
                    func.coalesce(
                        m.PipelineRun.last_heartbeat_at,
                        m.PipelineRun.started_at,
                    )
                    < threshold,
                )
            ).scalars().all()
            return [_orm_to_row(o, PipelineRunRow) for o in rows]

    def mark_failed_with_reason(
        self, *, run_id: UUID, agent: str, attempt: int, reason: str
    ) -> None:
        with self._sf() as s:
            s.execute(
                m.PipelineRun.__table__.update()
                .where(
                    m.PipelineRun.run_id == run_id,
                    m.PipelineRun.agent == agent,
                    m.PipelineRun.attempt == attempt,
                )
                .values(status="failed", error={"reason": reason})
            )

    def prior_statuses(self, *, run_id: UUID) -> dict[str, str]:
        with self._sf() as s:
            rows = s.execute(
                select(
                    m.PipelineRun.agent,
                    m.PipelineRun.status,
                    m.PipelineRun.attempt,
                )
                .where(m.PipelineRun.run_id == run_id)
                .order_by(m.PipelineRun.attempt.desc())
            ).all()
            result: dict[str, str] = {}
            for agent, status, _attempt in rows:
                # Highest-attempt status per agent wins.
                if agent not in result:
                    result[agent] = status
            return result

    @contextmanager
    def advisory_lock(self, *, key: int) -> Iterator[bool]:
        # Session-level advisory lock (NOT the *_xact_* variant): it is held on
        # the connection until explicitly unlocked, independent of transaction
        # boundaries, so it survives the commits the run performs on OTHER
        # pooled connections. We keep this one session open for the duration of
        # the `yield` so the lock is held across the whole run.
        with self._sf() as s:
            acquired = bool(
                s.execute(select(func.pg_try_advisory_lock(key))).scalar_one()
            )
            try:
                yield acquired
            finally:
                if acquired:
                    # Release explicitly — a pooled connection returned with the
                    # lock still held would never auto-release it. Suppress on a
                    # dead connection: backend exit releases the session lock.
                    with suppress(Exception):
                        s.execute(select(func.pg_advisory_unlock(key)))


class IngestWatermarkRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, *, source: str, scope: str) -> IngestWatermarkRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.IngestWatermark).where(
                    m.IngestWatermark.source == source,
                    m.IngestWatermark.scope == scope,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, IngestWatermarkRow) if o else None

    def upsert(self, row: IngestWatermarkRow) -> IngestWatermarkRow:
        _require_tz_aware(row.last_processed_at, "last_processed_at")
        values = _row_to_dict(row, drop_none_for=("updated_at",))
        with self._sf() as s:
            stmt = pg_insert(m.IngestWatermark).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["source", "scope"],
                set_={
                    "last_processed_at": stmt.excluded.last_processed_at,
                    "cursor": stmt.excluded.cursor,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.IngestWatermark).where(
                    m.IngestWatermark.source == row.source,
                    m.IngestWatermark.scope == row.scope,
                )
            ).scalar_one()
            return _orm_to_row(o, IngestWatermarkRow)


class DeadLetterRepositoryImpl:
    """append() must NEVER raise — the ingest path is already failing
    when it lands here and a second exception would crash the run.
    Every DB error is logged via structlog and swallowed.
    """

    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def append(self, row: DeadLetterRow) -> None:
        try:
            values = _row_to_dict(row, drop_none_for=("id", "created_at"))
            with self._sf() as s:
                orm = m.DeadLetter(**values)
                s.add(orm)
        except Exception as exc:
            # Logging itself may fail (broken handler, serialization error,
            # disk full). The never-raise contract means even a logging
            # failure must not escape — the ingest path is already in the
            # middle of handling an error when it landed here.
            try:
                _logger.error(
                    "dead_letter_append_failed",
                    error=type(exc).__name__,
                    error_message=str(exc),
                    source=row.source,
                    scope=row.scope,
                )
            except Exception:
                pass
            return

    def list_recent(self, *, limit: int = 100) -> Sequence[DeadLetterRow]:
        with self._sf() as s:
            rows = s.execute(
                select(m.DeadLetter)
                .order_by(m.DeadLetter.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [_orm_to_row(o, DeadLetterRow) for o in rows]


# ===========================================================================
# Player aliases (migration 008)
# ===========================================================================
class PlayerAliasRepositoryImpl:
    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def get(self, *, alias: str, source: str) -> PlayerAliasRow | None:
        with self._sf() as s:
            o = s.execute(
                select(m.PlayerAlias).where(
                    m.PlayerAlias.alias == alias,
                    m.PlayerAlias.source == source,
                )
            ).scalar_one_or_none()
            return _orm_to_row(o, PlayerAliasRow) if o else None

    def list_for_player(self, player_id: int) -> Sequence[PlayerAliasRow]:
        with self._sf() as s:
            rows = s.execute(
                select(m.PlayerAlias).where(m.PlayerAlias.player_id == player_id)
            ).scalars().all()
            return [_orm_to_row(o, PlayerAliasRow) for o in rows]

    def upsert(self, row: PlayerAliasRow) -> PlayerAliasRow:
        values = _row_to_dict(row, drop_none_for=_SERVER_MANAGED)
        with self._sf() as s:
            stmt = pg_insert(m.PlayerAlias).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["alias", "source"],
                set_={
                    "player_id": stmt.excluded.player_id,
                    "dob": stmt.excluded.dob,
                    "country_code": stmt.excluded.country_code,
                    "confidence": stmt.excluded.confidence,
                },
            )
            s.execute(stmt)
            s.flush()
            o = s.execute(
                select(m.PlayerAlias).where(
                    m.PlayerAlias.alias == row.alias,
                    m.PlayerAlias.source == row.source,
                )
            ).scalar_one()
            return _orm_to_row(o, PlayerAliasRow)


# ===========================================================================
# Elo snapshots (migration 011) — INSERT-only, append-only
# ===========================================================================
class EloSnapshotRepositoryImpl:
    """INSERT-only. Duplicate PK raises IdempotencyError — the contract
    is that the Research Agent never tries to re-write a snapshot it
    already wrote for the same (player, surface, match_id).
    """

    def __init__(self, session_factory: SessionCallable) -> None:
        self._sf = session_factory

    def insert(self, row: EloSnapshotRow) -> EloSnapshotRow:
        _require_tz_aware(row.as_of_ts, "as_of_ts")
        values = _row_to_dict(row, drop_none_for=("created_at",))
        try:
            with self._sf() as s:
                orm = m.EloSnapshot(**values)
                s.add(orm)
                s.flush()
                return _orm_to_row(orm, EloSnapshotRow)
        except IntegrityError as exc:
            # ONLY translate a duplicate-PK violation. Other integrity
            # failures (FK to a missing player or match, CHECK failures,
            # etc.) must propagate so the caller can investigate — silently
            # collapsing them to IdempotencyError would hide real corruption.
            if _is_unique_violation_on(exc, "elo_snapshots_pkey"):
                raise IdempotencyError(
                    f"elo_snapshot already exists for player_id={row.player_id} "
                    f"surface={row.surface!r} match_id={row.match_id}"
                ) from exc
            raise

    def get_latest_before(
        self,
        *,
        player_id: int,
        surface: EloSurface,
        as_of_ts: datetime,
    ) -> EloSnapshotRow | None:
        _require_tz_aware(as_of_ts, "as_of_ts")
        with self._sf() as s:
            o = s.execute(
                select(m.EloSnapshot)
                .where(
                    m.EloSnapshot.player_id == player_id,
                    m.EloSnapshot.surface == surface,
                    m.EloSnapshot.as_of_ts <= as_of_ts,
                )
                .order_by(m.EloSnapshot.as_of_ts.desc())
                .limit(1)
            ).scalar_one_or_none()
            return _orm_to_row(o, EloSnapshotRow) if o else None

    def career_match_counts(self) -> Mapping[int, int]:
        # COUNT(DISTINCT match_id) GROUP BY player_id. DISTINCT dedupes the two
        # rows (overall + surface) every Elo-updating match writes per player, so
        # the count equals the §M9 in-memory career counter the EloWalk produced.
        with self._sf() as s:
            rows = s.execute(
                select(
                    m.EloSnapshot.player_id,
                    func.count(func.distinct(m.EloSnapshot.match_id)),
                ).group_by(m.EloSnapshot.player_id)
            ).all()
            return {int(player_id): int(count) for player_id, count in rows}
