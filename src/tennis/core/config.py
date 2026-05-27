"""Typed config loader.

Loads `config/config.yaml` into a tree of pydantic models, then validates that
every required environment variable (named via `*_env` fields) is set for the
active environment. `validate_environment` is intended to be called once at
process startup — it fails loudly with a `MissingEnvironmentError` listing
every missing variable rather than blowing up later when an adapter is
actually used.

Secrets never live in the YAML. The YAML only stores the *name* of the env
var that holds the secret. Adapters read the secret via
`AppConfig.read_required_env(name)`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tennis.core.errors import InvalidConfigError, MissingEnvironmentError

Env = Literal["dev", "staging", "prod"]


# ---------------------------------------------------------------------------
# Env-dependency descriptor: each submodel declares which env vars it needs.
# `validate_environment` walks all submodels and collects them.
# ---------------------------------------------------------------------------
class EnvDep(BaseModel):
    name: str
    description: str
    required_in: tuple[Env, ...] = ("dev", "staging", "prod")

    def is_required(self, env: Env) -> bool:
        return env in self.required_in


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def env_deps(self) -> Iterator[EnvDep]:
        """Yield env vars this section depends on. Default: none."""
        return iter(())


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
class AppSection(_Section):
    env: Env = "dev"
    timezone: str = "UTC"
    run_id_prefix: str = "tennis"

    @model_validator(mode="after")
    def _utc_only(self) -> AppSection:
        if self.timezone != "UTC":
            raise InvalidConfigError("All timestamps must be UTC; set app.timezone: UTC")
        return self


class LoggingSection(_Section):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")
    redact_keys: tuple[str, ...] = ()


class DatabaseSection(_Section):
    url_env: str = "DATABASE_URL"
    pool_size: int = 10
    max_overflow: int = 5
    statement_timeout_ms: int = 30000
    migrations_dir: str = "migrations"

    def env_deps(self) -> Iterator[EnvDep]:
        yield EnvDep(
            name=self.url_env,
            description="Postgres connection URL (SQLAlchemy + psycopg)",
            required_in=("dev", "staging", "prod"),
        )


class QdrantCollectionConfig(_Section):
    vector_size: int
    distance: Literal["cosine", "euclidean", "dot"] = "cosine"
    hnsw_m: int = 32
    hnsw_ef_construct: int = 256
    payload_indexes: tuple[dict[str, str], ...] = ()


class QdrantSection(_Section):
    url_env: str = "QDRANT_URL"
    api_key_env: str = "QDRANT_API_KEY"
    api_key_required: bool = False
    collections: dict[str, QdrantCollectionConfig] = Field(default_factory=dict)

    def env_deps(self) -> Iterator[EnvDep]:
        yield EnvDep(
            name=self.url_env,
            description="Qdrant base URL",
            required_in=("staging", "prod"),
        )
        if self.api_key_required:
            yield EnvDep(
                name=self.api_key_env,
                description="Qdrant API key",
                required_in=("staging", "prod"),
            )


class EmbeddingsSection(_Section):
    provider: str
    model: str
    api_key_env: str
    dim: int
    batch_size: int = 64

    def env_deps(self) -> Iterator[EnvDep]:
        yield EnvDep(
            name=self.api_key_env,
            description=f"Embeddings API key ({self.provider})",
            required_in=("staging", "prod"),
        )


# --- sources ---------------------------------------------------------------
class SackmannSource(_Section):
    repo: str
    branch: str = "master"
    files: dict[str, str]
    cache_dir: str = ".cache/sackmann"
    local_mirror_dir: str = ".cache/sackmann/mirror"
    pin_commit_sha: str | None = None
    max_staleness_days: int = 3
    ranking_decades: tuple[str, ...] = ("90s", "00s", "10s", "20s")
    # H3: per-status authority for match_date when sources disagree.
    canonical_date_source: dict[str, str] = Field(
        default_factory=lambda: {
            "final": "sackmann",
            "scheduled": "atp_scraper",
            "live": "atp_scraper",
        }
    )


class AtpScraperSource(_Section):
    base_url: str
    user_agent: str
    user_agents: tuple[str, ...] = ()
    rotate_user_agent: bool = True
    request_timeout_s: int = 30
    rate_limit_rps: float = 0.5
    lookback_days: int = 21


class OpenWeatherSource(_Section):
    base_url: str
    api_key_env: str
    # H5: day_summary returns full-day aggregation; timemachine returns one
    # timestamp per call (24x cost for the same backfill).
    historical_endpoint: str = "/data/3.0/onecall/day_summary"
    forecast_endpoint: str = "/data/3.0/onecall"
    rate_limit_rps: float = 0.8
    max_forecast_horizon_h: int = 168
    on_missing_forecast: Literal["emit_null"] = "emit_null"

    def env_deps(self) -> Iterator[EnvDep]:
        yield EnvDep(
            name=self.api_key_env,
            description="OpenWeatherMap API key",
            required_in=("staging", "prod"),
        )


class OddsApiSource(_Section):
    base_url: str
    api_key_env: str
    sport: str
    bookmakers: tuple[str, ...]
    rate_limit_rps: float = 1.0
    closing_window_minutes: int = 15
    coverage_start_year: int = 2020   # H11: tennis_atp coverage begins ~2020
    match_linkage_window_days: int = 1  # J1: player+date match lookup window

    def env_deps(self) -> Iterator[EnvDep]:
        yield EnvDep(
            name=self.api_key_env,
            description="The Odds API key",
            required_in=("staging", "prod"),
        )


class SourcesSection(_Section):
    sackmann: SackmannSource
    atp_scraper: AtpScraperSource
    openweather: OpenWeatherSource
    odds_api: OddsApiSource

    def env_deps(self) -> Iterator[EnvDep]:
        yield from self.openweather.env_deps()
        yield from self.odds_api.env_deps()


# --- ingestion / features / modeling --------------------------------------
class IngestionRetry(_Section):
    max_attempts: int = 5
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0


class IntradayConflictConfig(_Section):
    enabled: bool = True
    log_to_run_metrics: bool = True


class SeasonRange(_Section):
    """H2: explicit `{start, end}` form; replaces ambiguous `[start, end]` list."""

    start: int
    end: int


class IngestionSection(_Section):
    history_backfill_season_range: SeasonRange
    daily_lookforward_days: int = 7
    retry: IngestionRetry = Field(default_factory=IngestionRetry)
    dead_letter_table: str = "dead_letter"
    intraday_conflict: IntradayConflictConfig = Field(default_factory=IntradayConflictConfig)


class EloConfig(_Section):
    # H10: variable K-factor + cold-start. k_base preserved for backward
    # compatibility but not used by the v1 Elo extractor.
    initial_rating: int = 1500
    k_new_player: int = 40
    k_threshold_matches: int = 30
    k_established: int = 20
    min_reliable_matches: int = 10
    k_base: float = 32.0
    surface_blend: float = 0.5
    glicko: bool = False
    materialize_snapshots: bool = True   # H7: elo_snapshots row per processed match


class MarketConfig(_Section):
    # H9: 'logit' removed — schema/CHECK is corrected in migration 012.
    devig_method_primary: Literal["shin", "proportional"] = "shin"
    devig_method_fallback: Literal["shin", "proportional"] = "proportional"
    use_closing_for_backtest: bool = True
    track_odds_drift_to_close: bool = True


class WeatherFeatureConfig(_Section):
    # H1: read by Modeling Agent only — feature_matrix always stores clean values.
    inject_forecast_noise: bool = True
    uncertainty_buckets: tuple[str, ...] = ("low", "medium", "high")
    # H4: forecast_horizon_h bands; missing observations -> NULL features.
    uncertainty_bucket_thresholds: dict[str, list[int]] = Field(default_factory=dict)
    noise_sigma_by_bucket: dict[str, dict[str, float]] = Field(default_factory=dict)
    max_obs_age_hours: int = 3


class DriftConfig(_Section):
    psi_threshold: float = 0.2
    ks_alpha: float = 0.01


class H2HConfig(_Section):
    # §M1: H2H confidence weighting + recency decay. Read by the H2H extractor
    # (R4); never hardcode these in the extractor. Both are DIVISORS in the §M1
    # math (recency-decay halflife; confidence full-sample), so they must be
    # strictly positive — guarded here (gt=0) to fail fast at config load,
    # mirroring the §M5 `live_decision_offset_hours` gt=0 precedent. A 0 or
    # negative value would otherwise crash or destabilise H2H extraction for the
    # whole batch at runtime.
    recency_decay_halflife_years: float = Field(default=2.0, gt=0.0)
    confidence_full_sample: int = Field(default=5, gt=0)


class SurfaceTransitionConfig(_Section):
    # §M2: surface transition exposure windows (days). Plain list default —
    # pydantic v2 copies mutable defaults per instance.
    adaptation_exposure_days: list[int] = [30, 90]


class ConditionsInteractionsConfig(_Section):
    # §M3: weather interaction allow-list. v1 = the two highest-signal terms;
    # all other interactions deferred.
    enabled: list[str] = ["wind_serve_risk", "altitude_serve_boost"]


class FeaturesSection(_Section):
    feature_set: str = "v1"
    windows_days: tuple[int, ...] = (7, 14, 30, 90, 365)
    elo: EloConfig = Field(default_factory=EloConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    weather: WeatherFeatureConfig = Field(default_factory=WeatherFeatureConfig)
    h2h: H2HConfig = Field(default_factory=H2HConfig)
    surface_transition: SurfaceTransitionConfig = Field(
        default_factory=SurfaceTransitionConfig
    )
    conditions_interactions: ConditionsInteractionsConfig = Field(
        default_factory=ConditionsInteractionsConfig
    )
    drift: DriftConfig = Field(default_factory=DriftConfig)


class SplitsConfig(_Section):
    method: Literal["walk_forward"] = "walk_forward"
    n_folds: int = 6
    # §M5: must be positive — a zero/negative warm-up makes every season
    # "testable" and breaks the walk-forward invariant; fail at config load.
    min_train_seasons: int = Field(default=8, gt=0)
    embargo_by: Literal["tournament_id"] = "tournament_id"


class XgbConfig(_Section):
    n_estimators: int = 800
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    early_stopping_rounds: int = 50
    n_jobs: int = -1


class LgbmConfig(_Section):
    n_estimators: int = 800
    num_leaves: int = 63
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    early_stopping_rounds: int = 50
    n_jobs: int = -1


class BaseLearnersConfig(_Section):
    xgb: XgbConfig = Field(default_factory=XgbConfig)
    lgbm: LgbmConfig = Field(default_factory=LgbmConfig)


class MetaLearnerConfig(_Section):
    type: Literal["logistic"] = "logistic"
    C: float = 1.0


class CalibrationConfig(_Section):
    method: Literal["platt", "isotonic"] = "platt"
    # §M21c: the tail is carved off the training data BEFORE walk-forward CV, so a
    # zero/negative value would degenerate the carve (and the calibrator) silently
    # — fail at config load instead (mirrors the §M5 live-offset gt=0 precedent).
    tail_days: int = Field(default=60, gt=0)
    min_calibration_samples: int = 50


class MetricsConfig(_Section):
    primary: str = "logloss"
    secondary: tuple[str, ...] = ("brier", "ece", "roi_kelly")


class EdgeConfig(_Section):
    min_edge_to_log: float = 0.01
    min_edge_to_bet: float = 0.03
    allow_missing_odds: bool = True


class KellyConfig(_Section):
    fraction: float = 0.25
    max_exposure_pct: float = 0.02
    max_total_exposure_pct: float = 0.10   # H12: same-day all-bets bankroll cap


class ModelingSection(_Section):
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    base_learners: BaseLearnersConfig = Field(default_factory=BaseLearnersConfig)
    meta_learner: MetaLearnerConfig = Field(default_factory=MetaLearnerConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    kelly: KellyConfig = Field(default_factory=KellyConfig)
    # §M20d: local directory the Modeling Agent joblib-serializes trained models
    # into (no blob store in v1). `artifact_uri` on `model_registry` is a path
    # under this dir. Relative paths resolve against the process working dir.
    # Must be non-empty — an empty dir would serialize artifacts to the CWD root.
    artifact_dir: str = Field(default="artifacts/models", min_length=1)
    # §M22: seed for the H1 training-noise RNG (np.random.default_rng). Training
    # noise is stochastic by design; a fixed seed makes each retrain — and the
    # unit suite — reproducible. Read by `models/noise.py` only.
    random_seed: int = 42


class DecisionTimingSection(_Section):
    # Must be positive: a zero/negative live offset makes pit_cut emit
    # as_of_ts >= start_ts (a lookahead). Fail at config load, not mid-run.
    live_decision_offset_hours: int = Field(default=24, gt=0)
    backtest_use_closing_line: bool = True


# --- briefing --------------------------------------------------------------
class LlmConfig(_Section):
    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2
    api_key_env: str

    def env_deps(self) -> Iterator[EnvDep]:
        yield EnvDep(
            name=self.api_key_env,
            description=f"{self.provider} API key",
            required_in=("staging", "prod"),
        )


class RagConfig(_Section):
    top_k: int = 8
    filters: dict[str, Any] = Field(default_factory=dict)
    prefer_human_edited_notes: bool = True


class SmtpConfig(_Section):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    host_env: str
    port: int = 587
    user_env: str
    password_env: str
    from_: str = Field(alias="from")
    tls: bool = True

    def env_deps(self) -> Iterator[EnvDep]:
        for name, desc in (
            (self.host_env, "SMTP host"),
            (self.user_env, "SMTP user"),
            (self.password_env, "SMTP password"),
        ):
            yield EnvDep(name=name, description=desc, required_in=("staging", "prod"))


class EmailConfig(_Section):
    smtp: SmtpConfig
    recipients_env: str
    subject_template: str = "Tennis edges — {date}"

    def env_deps(self) -> Iterator[EnvDep]:
        yield from self.smtp.env_deps()
        yield EnvDep(
            name=self.recipients_env,
            description="Comma-separated briefing recipients",
            required_in=("staging", "prod"),
        )


class BriefingSection(_Section):
    llm: LlmConfig
    rag: RagConfig = Field(default_factory=RagConfig)
    email: EmailConfig
    kelly_disclaimer: str = "For research purposes only. Not financial or betting advice."

    def env_deps(self) -> Iterator[EnvDep]:
        yield from self.llm.env_deps()
        yield from self.email.env_deps()


# --- orchestrator / monitor -----------------------------------------------
class HeartbeatConfig(_Section):
    interval_s: int = 30
    orphan_after_s: int = 300


class OrchestratorSection(_Section):
    cron: str = "30 6 * * *"
    steps_timeout_s: dict[str, int] = Field(default_factory=dict)
    on_failure: Literal["halt_and_alert", "continue_partial"] = "halt_and_alert"
    run_monitor_post_briefing: bool = True
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    orphan_sweep_on_start: bool = True


class MonitorSection(_Section):
    windows_days: tuple[int, ...] = (30, 60, 90)
    calibration: dict[str, float] = Field(default_factory=dict)
    drift: dict[str, float] = Field(default_factory=dict)
    output: dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy — tournament inclusion + match status filters
# ---------------------------------------------------------------------------
class TournamentPolicy(_Section):
    included_tiers: tuple[str, ...]
    excluded_categories: tuple[str, ...] = ()


class MatchStatusPolicy(_Section):
    training: tuple[str, ...]
    prediction: tuple[str, ...]


class PolicySection(_Section):
    tournaments: TournamentPolicy
    match_status: MatchStatusPolicy


# ---------------------------------------------------------------------------
# Player resolution
# ---------------------------------------------------------------------------
class PlayerResolutionSection(_Section):
    overrides_file: str = "config/player_overrides.yaml"
    fuzzy_threshold: float = 0.92
    require_dob_for_fuzzy: bool = True


# ---------------------------------------------------------------------------
# Feature engineering policy knobs
# ---------------------------------------------------------------------------
class MinWindowSamplesConfig(_Section):
    serve_return: int = 10
    elo_form: int = 5
    h2h: int = 1


class FeatureEngineeringSection(_Section):
    retirement_counts_as_match: bool = True
    walkover_counts_as_match: bool = False
    retirement_fatigue_weight: float = 0.5
    # H8: tier-keyed fallback for missing matches.best_of; pre-2000 Sackmann
    # rows frequently omit this. "default" is mandatory.
    best_of_null_fallback_by_tier: dict[str, int] = Field(default_factory=dict)
    min_window_samples: MinWindowSamplesConfig = Field(default_factory=MinWindowSamplesConfig)
    max_ranking_staleness_days: int = 7


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
class AppConfig(_Section):
    app: AppSection
    logging: LoggingSection
    database: DatabaseSection
    qdrant: QdrantSection
    embeddings: EmbeddingsSection
    sources: SourcesSection
    ingestion: IngestionSection
    features: FeaturesSection
    modeling: ModelingSection
    decision_timing: DecisionTimingSection
    briefing: BriefingSection
    orchestrator: OrchestratorSection
    monitor: MonitorSection
    policy: PolicySection
    player_resolution: PlayerResolutionSection
    feature_engineering: FeatureEngineeringSection

    def env_deps(self) -> Iterator[EnvDep]:
        for section in (
            self.database,
            self.qdrant,
            self.embeddings,
            self.sources,
            self.briefing,
        ):
            yield from section.env_deps()


# ---------------------------------------------------------------------------
# Loader + validator
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> AppConfig:
    """Load and validate `config/config.yaml`.

    Pure: does not touch the environment. Use `validate_environment` after
    loading to enforce env var presence.
    """
    p = Path(path)
    if not p.is_file():
        raise InvalidConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise InvalidConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic raises ValidationError; wrap with domain type
        raise InvalidConfigError(str(exc)) from exc


def required_env_for(config: AppConfig, env: Env | None = None) -> tuple[EnvDep, ...]:
    """Collect required env var descriptors for the given (or active) env."""
    target: Env = env if env is not None else config.app.env
    return tuple(dep for dep in config.env_deps() if dep.is_required(target))


def validate_environment(
    config: AppConfig,
    *,
    env: Env | None = None,
    environ: Sequence[tuple[str, str | None]] | None = None,
) -> None:
    """Fail loudly if any required env var is missing for the active env.

    `environ` is injectable for testing; defaults to `os.environ`.
    """
    target: Env = env if env is not None else config.app.env
    lookup = dict(environ) if environ is not None else dict(os.environ)
    missing = [
        dep.name
        for dep in required_env_for(config, target)
        if not (lookup.get(dep.name) or "").strip()
    ]
    if missing:
        raise MissingEnvironmentError(missing=missing, env=target)


def read_required_env(name: str) -> str:
    """Read an env var, raising MissingEnvironmentError if blank/missing.

    Use at the point of consumption (adapter construction). Startup validation
    is the first line of defense; this is the second.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingEnvironmentError(missing=[name], env=os.environ.get("APP_ENV", "unknown"))
    return value
