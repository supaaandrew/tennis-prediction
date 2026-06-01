"""Typed exception hierarchy for the entire project.

Every module raises a subclass of `TennisError`. Catch sites can narrow to the
specific subclass; logging/observability code can catch `TennisError` to attach
domain context without swallowing genuine programmer errors.
"""

from __future__ import annotations

from collections.abc import Sequence


class TennisError(Exception):
    """Base class for every domain exception in this project."""


# ---------------------------------------------------------------------------
# Config / startup
# ---------------------------------------------------------------------------
class ConfigError(TennisError):
    """Configuration could not be loaded or is internally inconsistent."""


class InvalidConfigError(ConfigError):
    """A config value is present but fails validation."""


class MissingEnvironmentError(ConfigError):
    """Required environment variables are not set.

    Raised by `core.config.validate_environment` at startup. Carries the list
    of missing variable names so the caller can log them all at once.
    """

    def __init__(self, missing: Sequence[str], env: str) -> None:
        self.missing: tuple[str, ...] = tuple(missing)
        self.env: str = env
        joined = ", ".join(self.missing)
        super().__init__(
            f"Missing required environment variables for env={env!r}: {joined}"
        )


# ---------------------------------------------------------------------------
# Ingestion / adapters
# ---------------------------------------------------------------------------
class IngestionError(TennisError):
    """Any failure during the Data Agent's ingest path."""


class SchemaValidationError(IngestionError):
    """A row from an external source failed pydantic validation."""


class AdapterError(IngestionError):
    """An external HTTP/scrape adapter could not return data."""


class RateLimitError(AdapterError):
    """Upstream returned a 429 or equivalent rate-limit signal."""


class UpstreamUnavailableError(AdapterError):
    """Upstream returned 5xx, timeout, or connection error."""


class MatchstatAuthError(AdapterError):
    """Matchstat (RapidAPI) rejected the credentials (HTTP 401/403).

    Raised by `MatchstatClient` when the API returns 401 or 403 — either the
    `X-RapidAPI-Key` is missing/invalid or the `X-RapidAPI-Host` header does
    not match the configured host (§T9). Retrying cannot help; the runbook
    diagnosis is "check `MATCHSTAT_API_KEY` and `sources.matchstat.host`".
    """


class MatchstatTierMismatchError(ConfigError):
    """Matchstat's `/ranks` lookup returned a name we don't recognize for one
    of the configured `tier_ids` (§T5 startup canary).

    The `config.sources.matchstat.tier_ids` tuple is fed verbatim into the
    fixtures `filter=TourRank:…` expression, so if matchstat ever renumbers
    its `Grand Slam` / `Masters 1000` / `ATP 500` / `ATP 250` ids the daily
    slate would silently miscategorize. Raised at DataAgent startup so the
    §L2 gate drops Data to `partial` instead.
    """


class MatchstatQuotaExhaustedError(AdapterError):
    """Matchstat free-tier monthly quota (500/month) has been exhausted (§T6).

    Raised at the 501st call within the current UTC calendar month. The
    counter is persisted via `IngestWatermarkRepository` under
    `(source='matchstat', scope='quota')`, so a process restart does not reset
    it. The slate adapter catches this at its boundary and converts it to
    `failures > 0` + `complete = False` so the §L2 gate degrades the run to
    `partial` instead of crashing.
    """


class SackmannStalenessError(IngestionError):
    """Local Sackmann mirror is older than `max_staleness_days`.

    Raised by the Data Agent on startup so the pipeline halts loudly rather
    than silently feeding stale historical data downstream.
    """

    def __init__(self, *, last_pulled_at: str, max_staleness_days: int) -> None:
        self.last_pulled_at = last_pulled_at
        self.max_staleness_days = max_staleness_days
        super().__init__(
            f"Sackmann mirror is stale: last_pulled_at={last_pulled_at}, "
            f"limit={max_staleness_days} days"
        )


class PlayerResolutionError(IngestionError):
    """A raw player reference could not be resolved to a canonical player_id.

    Either the candidate set was ambiguous (multiple matches above the fuzzy
    threshold and no DOB tiebreaker) or empty (no exact, fuzzy, or override
    match). The Data Agent should land the row in `dead_letter` rather than
    fabricate a player_id.
    """


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
class FeatureError(TennisError):
    """Failure in the Research Agent feature pipeline."""


class LookaheadViolationError(FeatureError):
    """A feature extractor referenced data past `as_of_ts`.

    This is the single most dangerous bug class in the project — if it fires,
    halt the pipeline. Carries the offending feature key and timestamps for
    forensic triage.
    """

    def __init__(self, *, feature_key: str, as_of_ts: str, accessed_ts: str) -> None:
        self.feature_key = feature_key
        self.as_of_ts = as_of_ts
        self.accessed_ts = accessed_ts
        super().__init__(
            f"Lookahead violation in feature {feature_key!r}: "
            f"as_of_ts={as_of_ts}, accessed_ts={accessed_ts}"
        )


class FeatureContractError(FeatureError):
    """A feature extractor returned values that violate its declared spec."""


class FeatureMatrixValidationError(FeatureError):
    """The Research→Modeling gate validator rejected the feature matrix.

    Carries the full list of violations so the operator sees every problem
    at once rather than one round-trip per violation.
    """

    def __init__(self, *, violations: Sequence[str]) -> None:
        self.violations: tuple[str, ...] = tuple(violations)
        if not violations:
            super().__init__("feature matrix validator failed (no violations recorded)")
        else:
            head, *rest = violations
            summary = head if not rest else f"{head} (+{len(rest)} more)"
            super().__init__(
                f"feature matrix validator rejected {len(violations)} row(s): {summary}"
            )


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
class ModelingError(TennisError):
    """Failure in train / calibrate / backtest."""


class InsufficientTrainingDataError(ModelingError):
    """Not enough labelled training rows / seasons to build a model.

    Raised by the Modeling Agent's data-assembly / split stage when the
    `for_training` slate cannot fill even one walk-forward fold after the
    `min_train_seasons` warm-up and the calibration-tail carve (§M21c). The
    agent maps this to a `failed` run with ZERO writes (no `model_registry`
    row), never a partial.
    """


class CalibrationError(ModelingError):
    """Calibrator could not be fit (e.g. degenerate held-out tail)."""


class BacktestError(ModelingError):
    """Backtest aborted mid-run."""


# ---------------------------------------------------------------------------
# Briefing
# ---------------------------------------------------------------------------
class BriefingError(TennisError):
    """Failure in the Briefing Agent (narrative generation or delivery)."""


class BriefingLlmError(BriefingError):
    """The LLM client could not return a narrative.

    Wraps any SDK/HTTP failure from the narrative provider. The cause is
    `redact_text`'d before it is logged/stored (§L10) because a transport
    error can carry the API key in a URL/message. A briefing LLM failure is
    NON-fatal (§N4): the email still sends without the narrative.
    """


class BriefingEmailError(BriefingError):
    """The email could not be delivered (SMTP connect/auth/send failure).

    Wraps any `smtplib`/transport failure. The cause is `redact_text`'d (§L10).
    An SMTP failure IS fatal for the run (§N4): nothing was delivered.
    """


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class StorageError(TennisError):
    """Database or vector-store IO failure."""


class IdempotencyError(StorageError):
    """An upsert produced a result inconsistent with idempotency guarantees."""


# ---------------------------------------------------------------------------
# Pipeline lineage / orchestration
# ---------------------------------------------------------------------------
class LineageError(TennisError):
    """Failure related to pipeline run lineage (preconditions, heartbeats)."""


class PreconditionNotMetError(LineageError):
    """An agent's declared precondition on a prior agent was not satisfied.

    Raised by `core.lineage.Precondition.check`. The agent must NOT proceed
    when this fires — silently running on stale upstream data is exactly the
    failure mode this exception exists to prevent.
    """

    def __init__(
        self,
        *,
        agent: str,
        expected: str,
        actual: str | None,
        run_id: str,
    ) -> None:
        self.agent = agent
        self.expected = expected
        self.actual = actual
        self.run_id = run_id
        super().__init__(
            f"precondition not met for run_id={run_id}: "
            f"expected agent={agent!r} status={expected!r}, got {actual!r}"
        )


class OrphanedRunError(LineageError):
    """A run row marked `running` has missed too many heartbeats.

    The orchestrator marks orphaned rows `failed` on the next cron trigger
    before starting a fresh run. Code paths that explicitly want to *check*
    for orphans (rather than just sweep them) raise this so the operator
    sees the offending run_id in their logs.
    """

    def __init__(self, *, run_id: str, agent: str, last_heartbeat_at: str | None) -> None:
        self.run_id = run_id
        self.agent = agent
        self.last_heartbeat_at = last_heartbeat_at
        super().__init__(
            f"orphaned run detected: run_id={run_id} agent={agent!r} "
            f"last_heartbeat_at={last_heartbeat_at!r}"
        )


class PipelineStartupError(LineageError):
    """The orchestrator could not even start a run — the database was
    unavailable while sweeping orphans or inserting the `running` row.

    No `pipeline_runs` row exists yet, so the outcome cannot be recorded in the
    table; the failure surfaces out-of-band (raised + logged) so the cron
    wrapper exits non-zero (§L2 'failed at startup').
    """
