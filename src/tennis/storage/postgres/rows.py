"""Row dataclasses — one frozen DTO per table in the Postgres schema.

These are pure value objects. No SQLAlchemy imports, no business logic, no
IO. Repositories accept and return them; the ORM mapping in `models.py` is
the bridge to actual database rows but the rest of the project never sees
it directly.

Schema source of truth is the migration files under `migrations/versions/`.
Every dataclass mirrors a table; field names match the SQL column names
exactly; Literal types mirror CHECK constraints.

Conventions:
  - Optional columns are typed as `T | None` with default `None`.
  - Server-managed columns (`created_at`, `updated_at`, BIGSERIAL ids) are
    `None`-defaulted so callers can construct rows for insert without the
    DB-populated values; reads from the DB carry them in.
  - JSONB fields are typed `tuple[...]` or `Mapping[str, Any]` depending
    on shape — kept immutable to honor `frozen=True`.
  - `aliases` on PlayerRow is the deprecated H6 column; readers may see
    legacy data, writers must NOT populate it (the resolver writes to
    `player_aliases` instead).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

# ---------------------------------------------------------------------------
# CHECK-constraint enums (mirror raw SQL in migrations)
# ---------------------------------------------------------------------------
Surface = Literal["Hard", "Clay", "Grass", "Carpet"]
EloSurface = Literal["Hard", "Clay", "Grass", "Carpet", "overall"]
Tier = Literal["GS", "Masters1000", "ATP500", "ATP250", "Challenger", "Futures", "Other"]
Round = Literal["R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "BR"]
MatchStatus = Literal["scheduled", "live", "final", "cancelled"]
DominantHand = Literal["R", "L", "U"]
Backhand = Literal["1", "2", "U"]
Dtype = Literal["float", "int", "bool", "cat"]
Confidence = Literal["exact", "fuzzy", "manual"]
RunStatusLit = Literal["running", "succeeded", "failed", "partial"]
DevigMethod = Literal["shin", "proportional"]
MatchDateSource = Literal["sackmann", "atp_scraper", "manual"]
Perspective = Literal["p1", "p2"]


# ---------------------------------------------------------------------------
# Migration 001 — core entities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PlayerRow:
    player_id: int
    full_name: str
    source: str
    source_uid: str
    country_code: str | None = None
    date_of_birth: date | None = None
    dominant_hand: DominantHand | None = None
    backhand: Backhand | None = None
    height_cm: int | None = None
    pro_since: int | None = None
    sackmann_atp_id: str | None = None
    aliases: tuple[Any, ...] = ()  # deprecated per H6
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PlayerRankingRow:
    player_id: int
    ranking_date: date
    rank: int
    points: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class VenueRow:
    venue_id: int
    city: str
    country_code: str
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TournamentRow:
    tournament_id: int
    season: int
    slug: str
    name: str
    tier: Tier
    surface: Surface
    indoor: bool
    draw_size: int | None = None
    venue_id: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 002 — matches + match_stats + odds_snapshots
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MatchRow:
    """One row in `matches`.

    `intraday_conflict` is audit-only per A10 — it is set by the Data Agent's
    post-ingest pass and never gates training or PIT.
    `match_date_source` (migration 012, H3) records which source supplied
    the canonical `match_date` used in the `match_id` hash.
    """

    match_id: int
    tournament_id: int
    round: Round
    match_date: date
    p1_id: int
    p2_id: int
    status: MatchStatus
    source: str
    source_uid: str
    start_ts: datetime | None = None
    winner_id: int | None = None
    loser_id: int | None = None
    score: str | None = None
    best_of: int | None = None
    sets_played: int | None = None
    minutes: int | None = None
    retired: bool = False
    walkover: bool = False
    intraday_conflict: bool = False
    match_date_source: MatchDateSource | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MatchStatRow:
    match_id: int
    player_id: int
    is_winner: bool
    aces: int | None = None
    double_faults: int | None = None
    serve_pts: int | None = None
    first_in: int | None = None
    first_won: int | None = None
    second_won: int | None = None
    serve_games: int | None = None
    bp_saved: int | None = None
    bp_faced: int | None = None
    winners: int | None = None
    unforced_errors: int | None = None
    net_points_won: int | None = None
    net_points_total: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class OddsSnapshotRow:
    """One snapshot of bookmaker pricing for one match.

    `devig_method` is narrowed to {'shin','proportional'} per H9 (migration
    012). Pre-H9 rows with 'logit' may still exist historically but the
    CHECK now rejects them on future writes; the DataAgent routes such
    payloads to `dead_letter`.
    """

    match_id: int
    bookmaker: str
    captured_at: datetime
    p1_decimal: float
    p2_decimal: float
    p1_implied: float
    p2_implied: float
    vig: float
    devig_method: DevigMethod
    market: str = "h2h"
    is_opening: bool = False
    is_closing: bool = False
    snapshot_id: int | None = None  # BIGSERIAL — server-assigned
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 003 — weather
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WeatherObservationRow:
    venue_id: int
    observed_at: datetime
    source: str
    is_forecast: bool
    temp_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_ms: float | None = None
    wind_dir_deg: int | None = None
    pressure_hpa: float | None = None
    precip_mm: float | None = None
    cloud_pct: int | None = None
    forecast_horizon_h: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class WeatherRevisionRow:
    venue_id: int
    observed_at: datetime
    source: str
    previous_row: Mapping[str, Any]
    new_row: Mapping[str, Any]
    revision_id: int | None = None  # BIGSERIAL
    revised_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 004 — features
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureSpecRow:
    feature_key: str
    version: int
    dtype: Dtype
    description: str | None = None
    formula_ref: str | None = None
    introduced_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class FeatureMatrixRow:
    """One per-(match, feature_set) feature payload.

    PK is `(match_id, feature_set)` per C3 — `perspective` is metadata, not
    part of the key. Trigger `fm_no_lookahead` enforces PIT on insert/update.
    """

    match_id: int
    feature_set: str
    as_of_ts: datetime
    payload: Mapping[str, Any]
    perspective: Perspective = "p1"
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 005 — predictions + model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelRegistryRow:
    version: str
    trained_at: datetime
    feature_set: str
    algo: str
    hyperparams: Mapping[str, Any]
    metrics: Mapping[str, Any]
    artifact_uri: str
    feature_hash: str
    data_window_start: date
    data_window_end: date
    is_active: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """One per-(match, model_version) calibrated probability + edges.

    `odds_drift_to_close` and `p1_implied_close` are backtest-only — they
    MUST be NULL in live prediction rows. Missing odds are allowed per
    C9: `edge_*` and `p1_implied_*` may all be None when no Pinnacle
    snapshot exists at decision time.
    """

    match_id: int
    model_version: str
    predicted_at: datetime
    p1_prob_raw: float
    p1_prob_cal: float
    p1_implied_open: float | None = None
    p1_implied_close: float | None = None
    p1_implied_decision: float | None = None
    edge_p1_shin: float | None = None
    edge_p2_shin: float | None = None
    edge_p1_proportional: float | None = None
    edge_p2_proportional: float | None = None
    kelly_fraction_p1: float | None = None
    kelly_fraction_p2: float | None = None
    odds_drift_to_close: float | None = None
    prediction_id: int | None = None  # BIGSERIAL
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BriefingDeliveryRow:
    """Email-delivery idempotency marker (migration 013, §N5/§S5).

    `UNIQUE(briefing_day_utc, model_version)` is the idempotency key; a row
    exists iff the day's briefing for that model was delivered. `run_id` is
    audit-only. `sent_at` is the confirmed-send instant (tz-aware UTC).
    """

    briefing_day_utc: date
    model_version: str
    run_id: UUID
    sent_at: datetime
    id: int | None = None  # BIGSERIAL
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 006 / 009 — runs + watermarks + dead_letter
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PipelineRunRow:
    run_id: UUID
    pipeline: str
    agent: str
    started_at: datetime
    status: RunStatusLit
    attempt: int = 1
    finished_at: datetime | None = None
    parent_run_id: UUID | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
    # Migration 009 additions
    last_heartbeat_at: datetime | None = None
    heartbeat_interval_s: int = 30


@dataclass(frozen=True, slots=True)
class IngestWatermarkRow:
    source: str
    scope: str
    last_processed_at: datetime
    cursor: Mapping[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DeadLetterRow:
    """A poison payload the DataAgent could not validate.

    `append` on the repository MUST swallow DB errors — losing a
    dead-letter row is preferable to crashing the ingest path that was
    already mid-failure when it tried to land here.
    """

    payload: Mapping[str, Any]
    error: Mapping[str, Any]
    run_id: UUID | None = None
    source: str | None = None
    scope: str | None = None
    id: int | None = None  # BIGSERIAL
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 008 — player aliases
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PlayerAliasRow:
    """Cross-source identity row. Per H6 this is the SOLE source of truth
    for alias resolution; `players.aliases` JSONB is deprecated.
    """

    alias: str
    source: str
    player_id: int
    confidence: Confidence
    dob: date | None = None
    country_code: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Migration 011 — elo snapshots
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EloSnapshotRow:
    """Append-only pre-match Elo snapshot.

    PK is `(player_id, surface, match_id)` per the audit fix — keying on
    `as_of_ts` collides when two same-day matches share `match_date - 1d`.
    `match_id` is NOT NULL; when a player has no snapshot the caller falls
    back to `config.features.elo.initial_rating` (1500 per H10).
    """

    player_id: int
    surface: EloSurface
    elo_rating: float
    as_of_ts: datetime
    match_id: int
    created_at: datetime | None = None
