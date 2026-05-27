"""DailyPipeline — owns the `pipeline_runs` lifecycle for one daily/training run.

Runs an ordered chain of agents under ONE `run_id` and ONE cluster-wide singleton
advisory lock (§S4). Per agent: precondition gate -> insert a `running` row ->
build a per-agent `AgentContext` (pinned clock + heartbeat closure) -> invoke the
agent -> map the result to a terminal status -> `update_status`. One orphan sweep
runs once, before the first agent.

Chain semantics (§S4):
  - Precondition gate: an agent whose precondition is not met for THIS run_id is
    SKIPPED — it writes no `pipeline_runs` row (there is no "skipped" status) and
    the loop continues. Its downstream gated agents then skip naturally; an agent
    with empty preconditions (DataAgent, MonitorAgent) always runs (Monitor A13).
  - An UNEXPECTED exception escaping `agent.run()` is recorded as that agent's
    terminal `failed` status and the loop CONTINUES (so Monitor still runs) — it
    is not re-raised. (Agents return an `AgentResult` for documented failures; an
    escape is a bug.)
  - `run_once()` returns the aggregate terminal status over the agents that RAN
    (`failed` if any ran-agent failed, else `partial` if any was partial, else
    `succeeded`), or `None` when another run already holds the lock.

Clock split (§L4): this class holds the REAL clock for run-lifecycle timestamps
(started_at, heartbeats, finished_at, orphan-sweep `now`). Each agent and every
adapter see a clock PINNED to `as_of` via `AgentContext.clock`. A frozen heartbeat
timestamp would make a live run look instantly orphaned, so the two clocks must
never be confused — the real clock never enters `AgentContext`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from uuid import UUID, uuid4

from tennis.core.clock import Clock, FrozenClock
from tennis.core.config import AppConfig
from tennis.core.contracts import Agent, AgentContext, AgentError, AgentResult
from tennis.core.errors import PipelineStartupError, PreconditionNotMetError
from tennis.core.lineage import RunStatus
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import PipelineRunRepository
from tennis.storage.postgres.rows import PipelineRunRow

_logger = get_logger("tennis.agents.orchestrator.pipeline")

_PIPELINE = "daily"
_ATTEMPT = 1
# Error codes that mean "nothing meaningful ran / was produced" -> 'failed'
# (§L2/§L3). `feature_matrix_invalid` is the Research C10 gate: a rejected matrix
# writes zero rows, so the run produced nothing usable downstream.
_FATAL_CODES = frozenset(
    {
        "staleness_halt",
        "preflight_error",
        "feature_matrix_invalid",
        # Modeling (M1): all mean "no usable model was produced / used" ->
        # 'failed' (not 'partial'), per AGENTS.md ModelingAgent status semantics.
        # `no_active_model` (M1b prediction mode) joins the set; the degraded
        # path `calibration_degraded` deliberately stays OUT (→ 'partial').
        "insufficient_training_data",
        "modeling_db_error",
        "no_active_model",
        # M1b prediction: the active model's feature family no longer matches the
        # config — the system is inconsistent, nothing usable can be scored.
        "feature_set_mismatch",
        # Briefing (B1, §N4): all mean "no email was delivered" -> 'failed'. The
        # `briefing_partial` code (some no-market rows / dead-lettered) stays OUT
        # of this set, so it maps to 'partial'.
        "no_qualifying_predictions",
        "smtp_send_failed",
        "briefing_db_error",
        # Monitor (§Q5): only a DB/read failure is fatal. The benign "not enough
        # data yet" codes `monitor_partial` / `monitor_no_active_model` stay OUT
        # of this set (→ 'partial') — the Monitor is observability and "no data
        # yet" is not an error (item 3).
        "monitor_db_error",
    }
)
# Fixed, arbitrary key for the cluster-wide singleton advisory lock guarding the
# daily run. Any stable constant works as long as it is unique among advisory-
# lock users in this database. Enforces the §L7 "no overlap" assumption in code
# rather than trusting the cron scheduler not to double-fire (Codex HIGH).
_DAILY_LOCK_KEY = 0x7A6E_6E69_5F64  # "tn_d" mnemonic; fits signed bigint


class DailyPipeline:
    """Run-row lifecycle owner for an ordered agent chain under one run_id."""

    def __init__(
        self,
        *,
        real_clock: Clock,
        runs: PipelineRunRepository,
        agents: Sequence[Agent],
        config: AppConfig,
        db: Any,
    ) -> None:
        self._real_clock = real_clock
        self._runs = runs
        self._agents = tuple(agents)
        self._config = config
        self._db = db

    def run_once(self) -> RunStatus | None:
        """Run the chain once under a cluster-wide singleton lock.

        Returns the aggregate terminal `RunStatus` over the agents that ran, or
        `None` when another run already holds the lock — in that case this
        invocation is a clean no-op (no row inserted, no orphan sweep), so an
        overlapping cron retry / manual trigger / >24h overrun cannot reap the
        live run's `running` row (§L7).
        """
        with self._singleton_lock() as acquired:
            if not acquired:
                _logger.warning("pipeline_already_running", lock_key=_DAILY_LOCK_KEY)
                return None
            return self._run_locked()

    def _run_locked(self) -> RunStatus:
        run_id = uuid4()
        as_of = self._real_clock.now()  # one pinned instant for the whole chain
        statuses: list[RunStatus] = []
        for i, agent in enumerate(self._agents):
            status = self._run_agent(
                agent, run_id=run_id, as_of=as_of, is_first=(i == 0)
            )
            if status is not None:  # None == skipped (precondition not met)
                statuses.append(status)
        aggregate = self._aggregate(statuses)
        _logger.info("pipeline_run_complete", run_id=str(run_id), status=aggregate)
        return aggregate

    def _run_agent(
        self, agent: Agent, *, run_id: UUID, as_of: Any, is_first: bool
    ) -> RunStatus | None:
        """Run one agent under the shared run_id. Returns its terminal status,
        or `None` if it was skipped because a precondition was not met (§S4)."""
        # Precondition gate (§S4): only read prior_statuses when the agent
        # actually declares preconditions — so the first agent (DataAgent, empty
        # preconditions) never issues a read before the §L2 startup guard.
        if agent.lineage.preconditions:
            try:
                agent.lineage.check_preconditions(
                    run_id=str(run_id),
                    prior_statuses=self._runs.prior_statuses(run_id=run_id),
                )
            except PreconditionNotMetError as exc:
                # Skip-and-continue: no pipeline_runs row (there is no "skipped"
                # status), a structured warning, and the loop moves on. Monitor
                # has empty preconditions, so it is never skipped here (A13).
                _logger.warning(
                    "pipeline_agent_skipped",
                    agent=agent.name,
                    cause=redact_text(str(exc)),
                )
                return None

        # §L2 startup guard: for the FIRST agent, the orphan sweep + its running
        # row are the first DB writes. If the DB is unavailable here, no row can
        # be written — there is no status to update, so it surfaces out-of-band.
        running_row = PipelineRunRow(
            run_id=run_id,
            # §L4: started_at is a run-lifecycle timestamp → the REAL clock at the
            # moment THIS agent's row is inserted, not the run's pinned `as_of`
            # (which the agent/adapters see). For a later stage this is hours
            # after `as_of`; using the real instant keeps the orphan-sweep
            # `COALESCE(last_heartbeat_at, started_at)` fallback honest per agent.
            pipeline=_PIPELINE,
            agent=agent.name,
            started_at=self._real_clock.now(),
            status="running",
            attempt=_ATTEMPT,
            heartbeat_interval_s=self._config.orchestrator.heartbeat.interval_s,
        )
        if is_first:
            try:
                self._sweep_orphans()
                self._runs.insert(running_row)
            except Exception as exc:
                _logger.error(
                    "pipeline_db_unavailable_at_startup", error=redact_text(repr(exc))
                )
                raise PipelineStartupError(
                    "database unavailable at pipeline startup"
                ) from exc
        else:
            self._runs.insert(running_row)

        ctx = AgentContext(
            run_id=run_id,
            as_of=as_of,
            config=self._config,
            db=self._db,
            clock=FrozenClock(as_of),  # pinned — every adapter's now() == as_of
            logger=_logger,
            heartbeat=self._make_heartbeat(run_id, agent.name),
        )
        ctx.heartbeat()  # initial beat right after the row exists
        try:
            result = agent.run(ctx)
        except Exception as exc:
            # An UNEXPECTED exception escaping run() (a bug, the §M9 single-shot-
            # rerun IdempotencyError, etc.) terminates THIS agent's row as
            # 'failed' (redacted, §L10) and the loop CONTINUES so Monitor still
            # runs (§S4). The terminal write keeps its §L8 propagate-on-failure
            # behavior — a failure there is the one case the orphan sweep reaps.
            _logger.error(
                "pipeline_agent_raised", agent=agent.name, error=redact_text(repr(exc))
            )
            self._runs.update_status(
                run_id=run_id,
                agent=agent.name,
                attempt=_ATTEMPT,
                status="failed",
                finished_at=self._real_clock.now(),
                error={"exception": f"{type(exc).__name__}: {redact_text(str(exc))}"},
            )
            return "failed"

        status = self._map_status(result)
        # §L8: a failure on this terminal write PROPAGATES — we never swallow it
        # or fabricate a status. The dangling 'running' row is reaped by the next
        # run's orphan sweep.
        self._runs.update_status(
            run_id=run_id,
            agent=agent.name,
            attempt=_ATTEMPT,
            status=status,
            finished_at=self._real_clock.now(),
            metrics=dict(result.metrics),
            error=self._serialize_errors(result.errors),
        )
        return status

    @staticmethod
    def _aggregate(statuses: Sequence[RunStatus]) -> RunStatus:
        """Aggregate the chain's per-agent terminal statuses (§S4/§S8): `failed`
        if any ran-agent failed, else `partial` if any was partial, else
        `succeeded`. An empty set (no agent ran — degenerate, all gated) is
        treated as `failed`."""
        if not statuses:
            return "failed"
        if "failed" in statuses:
            return "failed"
        if "partial" in statuses:
            return "partial"
        return "succeeded"

    # -- internals ----------------------------------------------------------
    @contextmanager
    def _singleton_lock(self) -> Iterator[bool]:
        """Hold the cluster-wide singleton advisory lock for the whole run so
        two overlapping invocations cannot run concurrently (and thus cannot
        orphan-sweep each other's live rows).

        A DB failure while ACQUIRING the lock is a startup failure — no row
        exists yet — so it is normalized to `PipelineStartupError` (§L2), the one
        typed contract for startup DB faults at the CLI/orchestrator boundary.
        Only the acquisition (`__enter__`) is wrapped; the run body's own
        exceptions propagate untouched (they own their per-agent terminal write).
        """
        cm = self._runs.advisory_lock(key=_DAILY_LOCK_KEY)
        try:
            acquired = cm.__enter__()
        except Exception as exc:
            _logger.error(
                "pipeline_db_unavailable_at_startup", error=redact_text(repr(exc))
            )
            raise PipelineStartupError(
                "database unavailable acquiring run lock"
            ) from exc
        try:
            yield acquired
        finally:
            cm.__exit__(None, None, None)

    def _make_heartbeat(self, run_id: UUID, agent_name: str) -> Callable[[], None]:
        """Build the per-agent emitter. Closes over the REAL clock so beats
        reflect wall time even though the agent only sees the pinned clock, and
        over the agent name so each stage beats its own row. Best-effort: a
        heartbeat hiccup must never abort a long ingest."""

        def _emit() -> None:
            try:
                self._runs.heartbeat(
                    run_id=run_id,
                    agent=agent_name,
                    attempt=_ATTEMPT,
                    now=self._real_clock.now(),
                )
            except Exception as exc:
                _logger.warning(
                    "pipeline_heartbeat_failed", error=redact_text(repr(exc))
                )

        return _emit

    def _sweep_orphans(self) -> None:
        """Mark crashed prior runs (status='running', no recent heartbeat) as
        failed before starting a fresh run (§L2)."""
        if not self._config.orchestrator.orphan_sweep_on_start:
            return
        now = self._real_clock.now()
        orphan_after_s = self._config.orchestrator.heartbeat.orphan_after_s
        for row in self._runs.orphans(orphan_after_s=orphan_after_s, now=now):
            self._runs.mark_failed_with_reason(
                run_id=row.run_id,
                agent=row.agent,
                attempt=row.attempt,
                reason="orphaned: no heartbeat within orphan_after_s",
            )
            _logger.warning(
                "pipeline_orphan_swept", run_id=str(row.run_id), agent=row.agent
            )

    @staticmethod
    def _map_status(result: AgentResult) -> RunStatus:
        codes = {e.code for e in result.errors}
        if codes & _FATAL_CODES:
            return "failed"
        return "succeeded" if result.ok else "partial"

    @staticmethod
    def _serialize_errors(errors: tuple[AgentError, ...]) -> dict[str, Any] | None:
        if not errors:
            return None
        return {
            "errors": [
                {"code": e.code, "message": e.message, "cause": e.cause}
                for e in errors
            ]
        }
