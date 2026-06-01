"""MonitorAgent — the post-pipeline observability stage (§Q, AGENTS.md A13).

NOT part of the 4-agent prediction chain (Data → Research → Modeling →
Briefing). The Monitor is a fifth `pipeline_runs` row that runs **regardless of
upstream status** (`lineage.preconditions = ()`, §Q7/A13); the only gate is
`config.orchestrator.run_monitor_post_briefing` at the orchestration layer, NOT
inside `run`.

It reads the day's-and-prior `predictions` for the ACTIVE model and computes,
per window in `config.monitor.windows_days`:

  * **ECE** — calibration error over SETTLED rows (reuse
    `models.metrics.expected_calibration_error`, 10 bins, §Q3).
  * **PSI** — drift of the `p1_prob_cal` distribution vs the immediately-
    preceding equal-width window (rolling reference, §Q4/§Q7).
  * **ROI** — fractional-Kelly realized return, settled at the fair
    `1/p1_implied_decision` odds (reuse `models.metrics.roi_kelly_backtest`,
    §Q2; documented no-vig optimism).

A prediction is SETTLED iff its match has gone `final` with a winner; the label
is `winner_id == p1_id` (§Q1). Not-yet-final predictions are PENDING — excluded
from ECE/ROI but COUNTED (`n_pending`).

Status (§Q5): `succeeded` = every window produced all three metrics; `partial` =
≥1 window had too few settled rows or a degraded metric (e.g. no reference
window yet → PSI `None`), or there is no active model to scope; `failed` = a DB
read failed (`monitor_db_error`, the only fatal code). The agent NEVER raises for
these documented modes — `run` returns an `AgentResult`. The orchestrator
persists `AgentResult.metrics` to `pipeline_runs.metrics` automatically
(`DailyPipeline._run_locked` → `update_status(metrics=…)`); the agent does not
write the run row itself.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

from tennis.core.config import AppConfig
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import StorageError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy
from tennis.core.logging import get_logger, redact_text

if TYPE_CHECKING:
    from tennis.adapters.matchstat.client import MatchstatClient
from tennis.models.metrics import (
    expected_calibration_error,
    population_stability_index,
    roi_kelly_backtest,
)
from tennis.storage.postgres.repositories import (
    MatchRepository,
    ModelRegistryRepository,
    PredictionRepository,
)
from tennis.storage.postgres.rows import MatchRow

_logger = get_logger("tennis.agents.monitor.agent")


class MonitorAgent:
    """Implements the `Agent` Protocol: `name`, `lineage`, `run(ctx)`."""

    name = "monitor"

    def __init__(
        self,
        *,
        config: AppConfig,
        model_registry_repo: ModelRegistryRepository,
        prediction_repo: PredictionRepository,
        match_repo: MatchRepository,
        matchstat_client: "MatchstatClient | None" = None,
    ) -> None:
        self._config = config
        self._model_registry_repo = model_registry_repo
        self._prediction_repo = prediction_repo
        self._match_repo = match_repo
        # §T6 — optional matchstat client for the quota-burn metric folded into
        # the §Q6 envelope. Best-effort: a failing read logs WARN and leaves the
        # envelope unchanged so monitor write is never gated by it.
        self._matchstat_client = matchstat_client

        hb = config.orchestrator.heartbeat
        # §Q7 / A13: the Monitor runs post-briefing REGARDLESS of upstream status,
        # so it declares NO preconditions. The run_monitor_post_briefing gate lives
        # in the orchestrator, never here.
        self.lineage = AgentLineage(
            preconditions=(),
            heartbeat=HeartbeatPolicy(
                interval_s=hb.interval_s, orphan_after_s=hb.orphan_after_s
            ),
        )

    # -- entrypoint ---------------------------------------------------------
    def run(self, ctx: AgentContext) -> AgentResult:
        try:
            return self._run(ctx)
        except StorageError as exc:
            cause = redact_text(str(exc))
            _logger.error("monitor_db_error", cause=cause)
            # Always emit the §Q6 envelope — even on the fatal path operators
            # inspect most. A downstream metrics reader sees the same fixed shape
            # (model_version null, windows empty) instead of a one-off payload.
            return AgentResult(
                ok=False,
                metrics=self._metrics_envelope(
                    None, ctx.as_of.astimezone(UTC), windows={}
                ),
                errors=(
                    AgentError(
                        code="monitor_db_error",
                        message="database unavailable during monitoring",
                        cause=cause,
                    ),
                ),
            )

    def _run(self, ctx: AgentContext) -> AgentResult:
        if ctx.as_of.tzinfo is None:  # mirrors §N3 _slate_window / AgentContext
            raise ValueError("MonitorAgent: as_of must be tz-aware")
        # Normalize to UTC before any window math: AgentContext guarantees
        # tz-aware but NOT UTC, and `as_of.date()` in a non-UTC zone could land
        # on a different calendar day than the pipeline's UTC contract, shifting
        # every window boundary (Codex F3). In production `as_of` is already UTC.
        as_of = ctx.as_of.astimezone(UTC)
        ctx.heartbeat()

        active = self._model_registry_repo.active()
        if active is None:
            # Nothing to scope metrics to yet — a benign "no data" state, not a
            # failure (item 3 ethos). Non-fatal code → `partial`.
            _logger.info("monitor_no_active_model")
            return AgentResult(
                ok=False,
                metrics=self._metrics_envelope(None, as_of, windows={}),
                errors=(
                    AgentError(
                        code="monitor_no_active_model",
                        message="no active model in the registry; nothing to monitor",
                    ),
                ),
            )

        windows: dict[str, dict[str, Any]] = {}
        for window_days in self._config.monitor.windows_days:
            ctx.heartbeat()
            windows[str(window_days)] = self._window_metrics(
                model_version=active.version,
                as_of=as_of,
                window_days=window_days,
            )

        metrics = self._metrics_envelope(active.version, as_of, windows=windows)
        complete = bool(windows) and all(
            _window_complete(w) for w in windows.values()
        )
        _logger.info(
            "monitor_complete",
            model_version=active.version,
            n_windows=len(windows),
            complete=complete,
        )
        if complete:
            return AgentResult(ok=True, metrics=metrics, errors=())
        return AgentResult(
            ok=False,
            metrics=metrics,
            errors=(
                AgentError(
                    code="monitor_partial",
                    message="one or more windows had insufficient data for a metric",
                ),
            ),
        )

    # -- per-window ---------------------------------------------------------
    def _window_metrics(
        self, *, model_version: str, as_of: datetime, window_days: int
    ) -> dict[str, Any]:
        """ECE/ROI over settled rows + rolling-reference PSI for one window."""
        cur_lo, cur_hi = _window_bounds(as_of, window_days, offset=0)
        ref_lo, ref_hi = _window_bounds(as_of, window_days, offset=1)

        current = self._prediction_repo.list_for_window(
            model_version=model_version, since=cur_lo, until=cur_hi
        )
        reference = self._prediction_repo.list_for_window(
            model_version=model_version, since=ref_lo, until=ref_hi
        )

        # ONE bulk match read for the whole window's outcome join, NOT one
        # `get` per prediction — a 90-day window is thousands of rows × 3 windows
        # (Codex F2, mirrors the §M18 anti-N+1 discipline). A DB failure here
        # raises a typed StorageError → run() → monitor_db_error (§Q5).
        matches = self._match_repo.list_for_ids(
            match_ids=[r.match_id for r in current]
        )

        probs: list[float] = []
        labels: list[int] = []
        implied: list[float | None] = []
        n_pending = 0
        for row in current:
            label = self._label(matches.get(row.match_id))
            if label is None:
                n_pending += 1
                continue
            probs.append(row.p1_prob_cal)
            labels.append(label)
            implied.append(row.p1_implied_decision)

        n_settled = len(labels)
        ece = (
            _nan_to_none(expected_calibration_error(labels, probs))
            if n_settled > 0
            else None
        )
        roi = (
            roi_kelly_backtest(
                probs=probs, y_true=labels, implied_p1=implied, config=self._config
            )
            if n_settled > 0
            else None
        )
        # PSI is settlement-independent — it measures OUTPUT drift, so it uses
        # EVERY prediction's calibrated prob in each window (§Q4). A reference
        # window with no rows (first-ever run) → NaN → None, never a crash.
        psi = _nan_to_none(
            population_stability_index(
                [r.p1_prob_cal for r in current],
                [r.p1_prob_cal for r in reference],
            )
        )
        return {
            "ece": ece,
            "psi": psi,
            "roi": roi,
            "n_settled": n_settled,
            "n_pending": n_pending,
        }

    @staticmethod
    def _label(match: MatchRow | None) -> int | None:
        """§Q1 realized-outcome join. `1` if p1 won, `0` if p2 won, `None` when
        the match is absent / not yet `final` / has no winner (pending) or the
        winner is neither listed player (a data anomaly — excluded, never
        guessed as a p2 win). `match` comes from the per-window bulk read."""
        if match is None or match.status != "final" or match.winner_id is None:
            return None
        if match.winner_id == match.p1_id:
            return 1
        if match.winner_id == match.p2_id:
            return 0
        _logger.warning(
            "monitor_winner_not_in_match",
            match_id=match.match_id,
            winner_id=match.winner_id,
        )
        return None

    def _metrics_envelope(
        self,
        model_version: str | None,
        as_of: datetime,
        *,
        windows: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "model_version": model_version,
            "as_of": as_of.isoformat(),
            "windows": windows,
        }
        # §T6 — best-effort matchstat quota-burn metric. Read BOTH properties
        # before mutating the envelope so a half-populated sub-dict can never
        # land if the second read raises. A failure logs WARN and leaves the
        # envelope unchanged — the §Q6 shape stays the same with or without it.
        if self._matchstat_client is not None:
            try:
                quota_remaining = self._matchstat_client.quota_remaining
                monthly_quota = self._matchstat_client.monthly_quota
                envelope["matchstat"] = {
                    "quota_remaining": quota_remaining,
                    "monthly_quota": monthly_quota,
                }
            except Exception as exc:  # noqa: BLE001 — never block monitor write
                _logger.warning(
                    "monitor_matchstat_metric_failed",
                    error=type(exc).__name__,
                )
        return envelope


# ---------------------------------------------------------------------------
# Pure helpers (module-level — no agent state)
# ---------------------------------------------------------------------------
def _window_bounds(
    as_of: datetime, window_days: int, *, offset: int
) -> tuple[datetime, datetime]:
    """Half-open `[lo, hi)` window of width `window_days`, anchored to the NEXT
    midnight UTC after the decisioning day (§N3 determinism — mirrors B1's
    `_slate_window`, which runs `[midnight(as_of), midnight(as_of)+1d)`).
    Anchoring on next-midnight makes the current window INCLUDE the run day's
    own predictions (written at ~06:30 UTC); a start-of-day midnight anchor would
    exclude them and leave the most-recent window empty on the day it runs (H1).
    `offset=0` is the current window `[anchor − D, anchor)`; `offset=1` is the
    immediately-preceding reference window `[anchor − 2D, anchor − D)` (§Q4
    rolling reference)."""
    anchor = datetime.combine(as_of.date(), time.min, tzinfo=UTC) + timedelta(days=1)
    hi = anchor - timedelta(days=window_days * offset)
    lo = hi - timedelta(days=window_days)
    return lo, hi


def _window_complete(window: dict[str, Any]) -> bool:
    """A window is fully populated iff all three metrics are present (non-None).
    A `None` on any metric (zero settled rows, or no reference window yet for
    PSI) makes the run `partial` (§Q5)."""
    return (
        window["ece"] is not None
        and window["psi"] is not None
        and window["roi"] is not None
    )


def _nan_to_none(value: float) -> float | None:
    return None if math.isnan(value) else value
