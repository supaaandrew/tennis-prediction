"""Repository Protocols.

Pure interfaces — no SQLAlchemy imports, no concrete implementations.
Agents depend on these Protocols; concrete implementations live in a
separate module that the DI container wires up at startup.

Naming convention: one Protocol per primary entity. Read methods return
Row dataclasses or None; write methods accept Row dataclasses and return
the persisted form (with server-managed fields populated).

Locked decisions baked into the interfaces (these are NOT optional knobs):
  - `MatchRepository.for_training()` returns only `status='final'` (C4).
  - `MatchRepository.for_prediction()` returns only
    `status IN ('scheduled','live')` (C4).
  - `EloSnapshotRepository.get_latest_before(...)` returns None when no
    snapshot exists; the caller (Research Agent) falls back to
    `config.features.elo.initial_rating` per H10 (1500).
  - `DeadLetterRepository.append(...)` MUST swallow any DB error — losing
    a dead-letter row is preferable to crashing the ingest path that was
    already in the middle of handling a failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from tennis.storage.postgres.rows import (
    BriefingDeliveryRow,
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


# ---------------------------------------------------------------------------
# Core entities (migration 001)
# ---------------------------------------------------------------------------
@runtime_checkable
class PlayerRepository(Protocol):
    def get(self, player_id: int) -> PlayerRow | None: ...
    def get_by_source(self, *, source: str, source_uid: str) -> PlayerRow | None: ...
    def upsert(self, row: PlayerRow) -> PlayerRow: ...


@runtime_checkable
class PlayerRankingRepository(Protocol):
    def get(self, *, player_id: int, ranking_date: date) -> PlayerRankingRow | None: ...
    def upsert(self, row: PlayerRankingRow) -> PlayerRankingRow: ...
    def latest_before(
        self, *, player_id: int, on_or_before: date
    ) -> PlayerRankingRow | None: ...


@runtime_checkable
class VenueRepository(Protocol):
    def get(self, venue_id: int) -> VenueRow | None: ...
    def get_by_city_country(
        self, *, city: str, country_code: str
    ) -> VenueRow | None: ...
    def list_all(self) -> Sequence[VenueRow]:
        """All venues ordered by `venue_id`. The DataAgent filters these for
        coordinate-bearing rows to feed the OWM adapter (§L5); in v1 the
        filtered list is empty until a geocoding pass populates lat/lon."""
        ...
    def upsert(self, row: VenueRow) -> VenueRow: ...


@runtime_checkable
class TournamentRepository(Protocol):
    def get(self, tournament_id: int) -> TournamentRow | None: ...
    def get_by_season_slug(
        self, *, season: int, slug: str
    ) -> TournamentRow | None: ...
    def upsert(self, row: TournamentRow) -> TournamentRow: ...


# ---------------------------------------------------------------------------
# Matches + stats + odds (migration 002, 012)
# ---------------------------------------------------------------------------
@runtime_checkable
class MatchRepository(Protocol):
    """Match persistence + the two policy-locked filters.

    `for_training` and `for_prediction` are NOT generic-status queries;
    their semantics are pinned by C4 / config.policy.match_status and
    must not accept a status parameter.
    """

    def get(self, match_id: int) -> MatchRow | None: ...
    def list_for_ids(
        self, *, match_ids: Sequence[int]
    ) -> Mapping[int, MatchRow]:
        """All matches among `match_ids`, keyed by `match_id` — the bulk dual of
        `get` (§Q). Empty `match_ids` → `{}` with NO DB round-trip. Lets the
        Monitor join a whole prediction window to realized outcomes in ONE read
        instead of one `get` per row (retires the per-row N+1; §M18 precedent).
        A DB/IO failure raises a typed `StorageError`."""
        ...

    def get_by_source(self, *, source: str, source_uid: str) -> MatchRow | None: ...
    def upsert(self, row: MatchRow) -> MatchRow:
        """Insert or merge a match, reconciling on the `match_id` PRIMARY KEY
        (K4) so a cross-source second write merges instead of colliding on the
        PK. `start_ts` is preserved via COALESCE — a NULL never clobbers a
        known intraday timestamp (H3/§15.2: `start_ts` is scraper-owned and
        Sackmann writes it NULL)."""
        ...

    def update_live_fields(
        self,
        *,
        match_id: int,
        start_ts: datetime | None,
        status: str,
        match_date_source: str,
    ) -> None:
        """Update ONLY the scraper-owned fields (`start_ts`, `status`,
        `match_date_source`) on an existing row, used by the ATP scraper's
        §K1 reconciliation path. No-op if `match_id` doesn't exist — log a
        warning, never raise."""
        ...

    def find_by_players_and_date(
        self,
        *,
        player_a_id: int,
        player_b_id: int,
        match_date: date,
        window_days: int = 1,
    ) -> MatchRow | None:
        """Match where the unordered pair `{p1_id, p2_id}` equals
        `{player_a_id, player_b_id}` and `match_date` is within
        `window_days` of the given date. Used by the Odds API adapter to
        link a bookmaker event (which carries no tournament/round) to an
        already-ingested match (J1).

        Returns the single matching row, or `None` when there is no match
        OR when more than one match is in range — ambiguity is never
        guessed; the caller dead-letters it (J1)."""
        ...

    def for_training(
        self, *, season_start: int, season_end: int
    ) -> Iterable[MatchRow]:
        """All matches with `status='final'` whose `match_date` falls in
        `[season_start, season_end]`. Status is fixed by C4."""
        ...

    def for_prediction(
        self, *, as_of: datetime, lookforward_days: int
    ) -> Iterable[MatchRow]:
        """All matches with `status IN ('scheduled','live')` whose
        `start_ts` (or `match_date` midnight UTC when start_ts NULL)
        falls in `(as_of, as_of + lookforward_days]`. Status set is
        fixed by C4."""
        ...

    def mark_intraday_conflict(self, *, match_id: int, conflict: bool) -> None:
        """Audit-only per A10; never gates training."""
        ...


@runtime_checkable
class MatchStatRepository(Protocol):
    def get(self, *, match_id: int, player_id: int) -> MatchStatRow | None: ...
    def list_for_match(self, match_id: int) -> Sequence[MatchStatRow]: ...
    def list_for_player(
        self, *, player_id: int, match_ids: Sequence[int]
    ) -> Mapping[int, MatchStatRow]: ...
    def upsert(self, row: MatchStatRow) -> MatchStatRow: ...


@runtime_checkable
class OddsSnapshotRepository(Protocol):
    def insert(self, row: OddsSnapshotRow) -> OddsSnapshotRow: ...
    def list_for_match(
        self, *, match_id: int, bookmaker: str | None = None
    ) -> Sequence[OddsSnapshotRow]: ...
    def opening(
        self, *, match_id: int, bookmaker: str, devig_method: DevigMethod
    ) -> OddsSnapshotRow | None: ...
    def closing(
        self, *, match_id: int, bookmaker: str, devig_method: DevigMethod
    ) -> OddsSnapshotRow | None: ...
    def latest_before(
        self,
        *,
        match_id: int,
        bookmaker: str,
        devig_method: DevigMethod,
        captured_before: datetime,
    ) -> OddsSnapshotRow | None:
        """The latest snapshot with `captured_at <= captured_before` for the
        (match, bookmaker, devig_method), or None. The boundary is INCLUSIVE per
        §15.4 — the decision-time feature uses `captured_at <= as_of_ts`, so a
        snapshot captured exactly at the PIT cut qualifies."""
        ...


# ---------------------------------------------------------------------------
# Weather (migration 003)
# ---------------------------------------------------------------------------
@runtime_checkable
class WeatherObservationRepository(Protocol):
    def get(
        self, *, venue_id: int, observed_at: datetime, source: str
    ) -> WeatherObservationRow | None: ...
    def upsert(
        self, row: WeatherObservationRow
    ) -> WeatherObservationRow: ...
    def nearest_at_or_before(
        self,
        *,
        venue_id: int,
        target_ts: datetime,
        source: str,
        max_age_hours: int,
    ) -> WeatherObservationRow | None:
        """Returns the latest observation with `observed_at <= target_ts`
        within `max_age_hours`. None if no observation qualifies — caller
        emits weather features as NULL per H4."""
        ...


@runtime_checkable
class WeatherRevisionRepository(Protocol):
    def append(self, row: WeatherRevisionRow) -> WeatherRevisionRow: ...


# ---------------------------------------------------------------------------
# Features (migration 004)
# ---------------------------------------------------------------------------
@runtime_checkable
class FeatureSpecRepository(Protocol):
    def get(self, *, feature_key: str, version: int) -> FeatureSpecRow | None: ...
    def list_active(self, *, feature_set: str) -> Sequence[FeatureSpecRow]: ...
    def upsert(self, row: FeatureSpecRow) -> FeatureSpecRow: ...


@runtime_checkable
class FeatureMatrixRepository(Protocol):
    def get(
        self, *, match_id: int, feature_set: str
    ) -> FeatureMatrixRow | None: ...
    def upsert(self, row: FeatureMatrixRow) -> FeatureMatrixRow:
        """Trigger `fm_no_lookahead` enforces PIT on every write — a
        violation raises a DB-side exception which the caller MUST NOT
        catch silently."""
        ...

    def list_for_matches(
        self, *, match_ids: Sequence[int], feature_set: str
    ) -> Sequence[FeatureMatrixRow]: ...


# ---------------------------------------------------------------------------
# Predictions + model registry (migration 005)
# ---------------------------------------------------------------------------
@runtime_checkable
class ModelRegistryRepository(Protocol):
    def get(self, version: str) -> ModelRegistryRow | None: ...
    def active(self) -> ModelRegistryRow | None:
        """Returns the single row with `is_active=TRUE`, or None. The
        partial unique index in migration 005 enforces ≤1 active."""
        ...

    def insert(self, row: ModelRegistryRow) -> ModelRegistryRow: ...
    def activate(self, version: str) -> None: ...


@runtime_checkable
class PredictionRepository(Protocol):
    def get(
        self, *, match_id: int, model_version: str
    ) -> PredictionRow | None: ...
    def upsert(self, row: PredictionRow) -> PredictionRow: ...
    def list_for_window(
        self, *, model_version: str, since: datetime, until: datetime
    ) -> Sequence[PredictionRow]: ...


@runtime_checkable
class BriefingDeliveryRepository(Protocol):
    """Email-delivery idempotency (§N5/§S5). The BriefingAgent `get`s before
    send (skip if a row exists) and `record`s after a confirmed send."""

    def get(
        self, *, briefing_day_utc: date, model_version: str
    ) -> BriefingDeliveryRow | None: ...

    def record(self, row: BriefingDeliveryRow) -> None:
        """Insert the delivery marker; a duplicate `(briefing_day_utc,
        model_version)` is a no-op (INSERT … ON CONFLICT DO NOTHING). A DB/IO
        failure raises a TYPED `StorageError`."""
        ...


# ---------------------------------------------------------------------------
# Pipeline lineage (migration 006, 009)
# ---------------------------------------------------------------------------
@runtime_checkable
class PipelineRunRepository(Protocol):
    def get(
        self, *, run_id: UUID, agent: str, attempt: int = 1
    ) -> PipelineRunRow | None: ...
    def insert(self, row: PipelineRunRow) -> PipelineRunRow: ...
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
    ) -> None: ...
    def heartbeat(
        self, *, run_id: UUID, agent: str, attempt: int, now: datetime
    ) -> None:
        """Update `last_heartbeat_at = now` on the (run, agent, attempt)
        row. Idempotent."""
        ...

    def orphans(self, *, orphan_after_s: int, now: datetime) -> Sequence[PipelineRunRow]:
        """Rows in status='running' that have not heartbeat within
        `orphan_after_s` seconds. Uses `started_at` as the reference
        when `last_heartbeat_at IS NULL` (run crashed pre-first-beat)."""
        ...

    def mark_failed_with_reason(
        self, *, run_id: UUID, agent: str, attempt: int, reason: str
    ) -> None:
        """Used by the orchestrator's orphan-sweep."""
        ...

    def prior_statuses(self, *, run_id: UUID) -> dict[str, str]:
        """Map of agent name → latest terminal status for this run_id.
        Fed to `Precondition.check` in `core.lineage`."""
        ...

    def advisory_lock(self, *, key: int) -> AbstractContextManager[bool]:
        """Cluster-wide Postgres session advisory lock for singleton execution.

        The returned context manager yields ``True`` if the lock was acquired
        (the caller may proceed) or ``False`` if another session already holds
        it (the caller must abort). The lock is held for the duration of the
        context and released on exit. DailyPipeline takes this before sweeping
        orphans so two overlapping invocations cannot reap each other's live
        `running` rows."""
        ...


@runtime_checkable
class IngestWatermarkRepository(Protocol):
    def get(self, *, source: str, scope: str) -> IngestWatermarkRow | None: ...
    def upsert(self, row: IngestWatermarkRow) -> IngestWatermarkRow: ...


@runtime_checkable
class DeadLetterRepository(Protocol):
    def append(self, row: DeadLetterRow) -> None:
        """MUST swallow any DB exception. The ingest path is already
        failing when it lands here; raising again would suppress the
        original error and crash the run. Implementations should log
        the failure via structlog and return normally."""
        ...

    def list_recent(self, *, limit: int = 100) -> Sequence[DeadLetterRow]: ...


# ---------------------------------------------------------------------------
# Player aliases (migration 008)
# ---------------------------------------------------------------------------
@runtime_checkable
class PlayerAliasRepository(Protocol):
    """Sole source of truth for cross-source alias resolution per H6.
    `players.aliases` JSONB is deprecated and must not be written.
    """

    def get(self, *, alias: str, source: str) -> PlayerAliasRow | None: ...
    def list_for_player(self, player_id: int) -> Sequence[PlayerAliasRow]: ...
    def upsert(self, row: PlayerAliasRow) -> PlayerAliasRow: ...


# ---------------------------------------------------------------------------
# Elo snapshots (migration 011)
# ---------------------------------------------------------------------------
@runtime_checkable
class EloSnapshotRepository(Protocol):
    """Append-only. `insert` is the only write; the PIT-safe read pattern
    is `get_latest_before`.
    """

    def insert(self, row: EloSnapshotRow) -> EloSnapshotRow: ...

    def get_latest_before(
        self,
        *,
        player_id: int,
        surface: EloSurface,
        as_of_ts: datetime,
    ) -> EloSnapshotRow | None:
        """Latest snapshot with `as_of_ts <= as_of_ts` for the given
        (player, surface). Returns None when no snapshot exists — the
        caller MUST fall back to `config.features.elo.initial_rating`
        (1500 per H10) rather than raise."""
        ...

    def career_match_counts(self) -> Mapping[int, int]:
        """Per-player career Elo-updating-match count, reconstructed from the
        built ladder as `COUNT(DISTINCT match_id)` grouped by `player_id`.

        This is the PREDICTION-path source for the §M9 career counter — which is
        in-memory only during the training `EloWalk` — that drives the H10
        K-factor switch and `reliability_low`. Exact: a walkover writes NO
        snapshot (§M10) and each Elo-updating match writes two rows per player
        (overall + surface), so the distinct-`match_id` count per player equals
        the walk's counter. Returns `{}` for an empty ladder; a player absent
        from the map has 0 (the `EloExtractor` defaults a missing key to 0)."""
        ...
