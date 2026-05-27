"""SQLAlchemy 2.0 declarative models.

One ORM class per table in the Postgres schema. These are the ONLY place
in the project where SQLAlchemy types appear — `rows.py` is pure data,
`repositories.py` is pure interface. Concrete repository implementations
(future work) import from this module to talk to the DB.

The migrations are authoritative for the schema; these models are reflected
from them by hand. Triggers, partial indexes, and CHECK constraints live
exclusively in the migrations — SQLAlchemy `autogenerate` is disabled
(see `migrations/env.py`).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CHAR,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, REAL, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base. Concrete repositories use this for ORM-style ops."""


# ---------------------------------------------------------------------------
# Migration 001 — core entities
# ---------------------------------------------------------------------------
class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    country_code: Mapped[str | None] = mapped_column(CHAR(3))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    dominant_hand: Mapped[str | None] = mapped_column(CHAR(1))
    backhand: Mapped[str | None] = mapped_column(CHAR(1))
    height_cm: Mapped[int | None] = mapped_column(SmallInteger)
    pro_since: Mapped[int | None] = mapped_column(SmallInteger)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_uid: Mapped[str] = mapped_column(Text, nullable=False)
    sackmann_atp_id: Mapped[str | None] = mapped_column(Text)
    # Deprecated per H6; kept for legacy reads only.
    aliases: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint("source", "source_uid", name="players_source_source_uid_key"),
    )


class PlayerRanking(Base):
    __tablename__ = "player_rankings"

    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    ranking_date: Mapped[date] = mapped_column(Date, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    points: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Venue(Base):
    __tablename__ = "venues"

    venue_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    city: Mapped[str] = mapped_column(Text, nullable=False)
    country_code: Mapped[str] = mapped_column(CHAR(3), nullable=False)
    latitude: Mapped[float | None] = mapped_column(REAL)
    longitude: Mapped[float | None] = mapped_column(REAL)
    altitude_m: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint("city", "country_code", name="venues_city_country_key"),
    )


class Tournament(Base):
    __tablename__ = "tournaments"

    tournament_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    season: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    surface: Mapped[str] = mapped_column(Text, nullable=False)
    indoor: Mapped[bool] = mapped_column(Boolean, nullable=False)
    draw_size: Mapped[int | None] = mapped_column(SmallInteger)
    venue_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("venues.venue_id")
    )
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint("season", "slug", name="tournaments_season_slug_key"),
    )


# ---------------------------------------------------------------------------
# Migration 002 / 012 — matches, match_stats, odds_snapshots
# ---------------------------------------------------------------------------
class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tournament_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tournaments.tournament_id"), nullable=False
    )
    round: Mapped[str] = mapped_column(Text, nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    start_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    winner_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("players.player_id")
    )
    loser_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("players.player_id")
    )
    p1_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.player_id"), nullable=False
    )
    p2_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.player_id"), nullable=False
    )
    score: Mapped[str | None] = mapped_column(Text)
    best_of: Mapped[int | None] = mapped_column(SmallInteger)
    sets_played: Mapped[int | None] = mapped_column(SmallInteger)
    minutes: Mapped[int | None] = mapped_column(Integer)
    retired: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    walkover: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    intraday_conflict: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    # Migration 012, H3 — nullable; populated by DataAgent on inserts.
    match_date_source: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_uid: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint("source", "source_uid", name="matches_source_source_uid_key"),
    )


class MatchStat(Base):
    __tablename__ = "match_stats"

    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True
    )
    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.player_id"), primary_key=True
    )
    is_winner: Mapped[bool] = mapped_column(Boolean, nullable=False)
    aces: Mapped[int | None] = mapped_column(SmallInteger)
    double_faults: Mapped[int | None] = mapped_column(SmallInteger)
    serve_pts: Mapped[int | None] = mapped_column(SmallInteger)
    first_in: Mapped[int | None] = mapped_column(SmallInteger)
    first_won: Mapped[int | None] = mapped_column(SmallInteger)
    second_won: Mapped[int | None] = mapped_column(SmallInteger)
    serve_games: Mapped[int | None] = mapped_column(SmallInteger)
    bp_saved: Mapped[int | None] = mapped_column(SmallInteger)
    bp_faced: Mapped[int | None] = mapped_column(SmallInteger)
    winners: Mapped[int | None] = mapped_column(SmallInteger)
    unforced_errors: Mapped[int | None] = mapped_column(SmallInteger)
    net_points_won: Mapped[int | None] = mapped_column(SmallInteger)
    net_points_total: Mapped[int | None] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    snapshot_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False
    )
    bookmaker: Mapped[str] = mapped_column(Text, nullable=False)
    market: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'h2h'"))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_opening: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    is_closing: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    p1_decimal: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    p2_decimal: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    p1_implied: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    p2_implied: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    vig: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    devig_method: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "bookmaker",
            "market",
            "captured_at",
            "devig_method",
            name="odds_snapshots_dedup_key",
        ),
    )


# ---------------------------------------------------------------------------
# Migration 003 — weather
# ---------------------------------------------------------------------------
class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    venue_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("venues.venue_id"), primary_key=True
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    source: Mapped[str] = mapped_column(Text, primary_key=True)
    is_forecast: Mapped[bool] = mapped_column(Boolean, nullable=False)
    temp_c: Mapped[float | None] = mapped_column(REAL)
    humidity_pct: Mapped[float | None] = mapped_column(REAL)
    wind_speed_ms: Mapped[float | None] = mapped_column(REAL)
    wind_dir_deg: Mapped[int | None] = mapped_column(SmallInteger)
    pressure_hpa: Mapped[float | None] = mapped_column(REAL)
    precip_mm: Mapped[float | None] = mapped_column(REAL)
    cloud_pct: Mapped[int | None] = mapped_column(SmallInteger)
    forecast_horizon_h: Mapped[int | None] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class WeatherRevision(Base):
    __tablename__ = "weather_revisions"

    revision_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    venue_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("venues.venue_id"), nullable=False
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    previous_row: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    new_row: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    revised_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------
# Migration 004 — features
# ---------------------------------------------------------------------------
class FeatureSpec(Base):
    __tablename__ = "feature_specs"

    feature_key: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    dtype: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    formula_ref: Mapped[str | None] = mapped_column(Text)
    introduced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class FeatureMatrix(Base):
    __tablename__ = "feature_matrix"

    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True
    )
    feature_set: Mapped[str] = mapped_column(Text, primary_key=True)
    perspective: Mapped[str] = mapped_column(
        CHAR(2), nullable=False, server_default=text("'p1'")
    )
    as_of_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------
# Migration 005 — model registry + predictions
# ---------------------------------------------------------------------------
class ModelRegistry(Base):
    __tablename__ = "model_registry"

    version: Mapped[str] = mapped_column(Text, primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_set: Mapped[str] = mapped_column(Text, nullable=False)
    algo: Mapped[str] = mapped_column(Text, nullable=False)
    hyperparams: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    feature_hash: Mapped[str] = mapped_column(Text, nullable=False)
    data_window_start: Mapped[date] = mapped_column(Date, nullable=False)
    data_window_end: Mapped[date] = mapped_column(Date, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(
        Text, ForeignKey("model_registry.version"), nullable=False
    )
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    p1_prob_raw: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    p1_prob_cal: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    p1_implied_open: Mapped[float | None] = mapped_column(Numeric(8, 6))
    p1_implied_close: Mapped[float | None] = mapped_column(Numeric(8, 6))
    p1_implied_decision: Mapped[float | None] = mapped_column(Numeric(8, 6))
    edge_p1_shin: Mapped[float | None] = mapped_column(Numeric(9, 6))
    edge_p2_shin: Mapped[float | None] = mapped_column(Numeric(9, 6))
    edge_p1_proportional: Mapped[float | None] = mapped_column(Numeric(9, 6))
    edge_p2_proportional: Mapped[float | None] = mapped_column(Numeric(9, 6))
    kelly_fraction_p1: Mapped[float | None] = mapped_column(Numeric(8, 6))
    kelly_fraction_p2: Mapped[float | None] = mapped_column(Numeric(8, 6))
    odds_drift_to_close: Mapped[float | None] = mapped_column(Numeric(9, 6))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint("match_id", "model_version", name="predictions_match_model_key"),
    )


class BriefingDelivery(Base):
    """Email-delivery idempotency marker (migration 013, §N5/§S5).

    One row per delivered briefing. `UNIQUE(briefing_day_utc, model_version)`
    is the idempotency key; `run_id` is audit-only.
    """

    __tablename__ = "briefing_deliveries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    briefing_day_utc: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(
        Text, ForeignKey("model_registry.version"), nullable=False
    )
    run_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint(
            "briefing_day_utc", "model_version", name="briefing_deliveries_day_model_key"
        ),
    )


# ---------------------------------------------------------------------------
# Migration 006 / 009 — pipeline_runs + ingest_watermarks + dead_letter
# ---------------------------------------------------------------------------
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    pipeline: Mapped[str] = mapped_column(Text, nullable=False)
    agent: Mapped[str] = mapped_column(Text, primary_key=True)
    attempt: Mapped[int] = mapped_column(
        SmallInteger, primary_key=True, server_default=text("1")
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    parent_run_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Migration 009 additions.
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_interval_s: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("30")
    )


class IngestWatermark(Base):
    __tablename__ = "ingest_watermarks"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    scope: Mapped[str] = mapped_column(Text, primary_key=True)
    last_processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cursor: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class DeadLetter(Base):
    __tablename__ = "dead_letter"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    source: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    error: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------
# Migration 008 — player aliases
# ---------------------------------------------------------------------------
class PlayerAlias(Base):
    __tablename__ = "player_aliases"

    alias: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str] = mapped_column(Text, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.player_id", ondelete="CASCADE"), nullable=False
    )
    dob: Mapped[date | None] = mapped_column(Date)
    country_code: Mapped[str | None] = mapped_column(CHAR(3))
    confidence: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------
# Migration 011 — elo snapshots
# ---------------------------------------------------------------------------
class EloSnapshot(Base):
    __tablename__ = "elo_snapshots"

    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True
    )
    surface: Mapped[str] = mapped_column(Text, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True
    )
    elo_rating: Mapped[float] = mapped_column(REAL, nullable=False)
    as_of_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        Index(
            "elo_snapshots_lookup_idx_orm",
            "player_id",
            "surface",
            "as_of_ts",
        ),
    )
