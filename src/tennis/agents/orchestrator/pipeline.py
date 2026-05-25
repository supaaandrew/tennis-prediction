"""DailyPipeline — owns the `pipeline_runs` lifecycle for the daily ingest (P7).

orphan-sweep -> insert a `running` row -> build a per-run `AgentContext` (pinned
clock + heartbeat closure) -> invoke the agent -> map the result to a terminal
status -> `update_status`. The DataAgent does the ingest; this class does the
bookkeeping and is the seam the other three agents will plug into later.

Clock split (§L4): this class holds the REAL clock for run-lifecycle timestamps
(started_at, heartbeats, finished_at, orphan-sweep `now`). The agent and every
adapter see a clock PINNED to `as_of` via `AgentContext.clock`. A frozen heartbeat
timestamp would make a live run look instantly orphaned, so the two clocks must
never be confused — the real clock never enters `AgentContext`.

v1 scope: `run_once()` invokes the DataAgent only. The downstream precondition
chain (Research -> Modeling -> Briefing) is future work; the gate is a no-op stub.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID, uuid4

from tennis.core.clock import Clock, FrozenClock
from tennis.core.config import AppConfig
from tennis.core.contracts import Agent, AgentContext, AgentError, AgentResult
from tennis.core.errors import PipelineStartupError
from tennis.core.lineage import RunStatus
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import PipelineRunRepository
from tennis.storage.postgres.rows import PipelineRunRow

_logger = get_logger("tennis.agents.orchestrator.pipeline")

_PIPELINE = "daily"
_ATTEMPT = 1
# Error codes that mean "nothing meaningful ran" -> 'failed' (§L2/§L3).
_FATAL_CODES = frozenset({"staleness_halt", "preflight_error"})
# Fixed, arbitrary key for the cluster-wide singleton advisory lock guarding the
# daily run. Any stable constant works as long as it is unique among advisory-
# lock users in this database. Enforces the §L7 "no overlap" assumption in code
# rather than trusting the cron scheduler not to double-fire (Codex HIGH).
_DAILY_LOCK_KEY = 0x7A6E_6E69_5F64  # "tn_d" mnemonic; fits signed bigint


class DailyPipeline:
    """Run-row lifecycle owner for the daily DataAgent run."""

    def __init__(
        self,
        *,
        real_clock: Clock,
        runs: PipelineRunRepository,
        agent: Agent,
        config: AppConfig,
        db: Any,
    ) -> None:
        self._real_clock = real_clock
        self._runs = runs
        self._agent = agent
        self._config = config
        self._db = db

    def run_once(self) -> RunStatus | None:
        """Run the daily pipeline once under a cluster-wide singleton lock.

        Returns the terminal `RunStatus`, or `None` when another run already
        holds the lock — in that case this invocation is a clean no-op (no row
        inserted, no orphan sweep), so an overlapping cron retry / manual
        trigger / >24h overrun cannot reap the live run's `running` row (§L7).
        """
        with self._singleton_lock() as acquired:
            if not acquired:
                _logger.warning("pipeline_already_running", lock_key=_DAILY_LOCK_KEY)
                return None
            return self._run_locked()

    def _run_locked(self) -> RunStatus:
        run_id = uuid4()
        as_of = self._real_clock.now()

        # DB-startup guard (§L2): orphans()/insert() are the first DB writes.
        # If the DB is unavailable here, no row can be written — there is no
        # status to update, so the failure surfaces out-of-band.
        try:
            self._sweep_orphans()
            self._runs.insert(
                PipelineRunRow(
                    run_id=run_id,
                    pipeline=_PIPELINE,
                    agent=self._agent.name,
                    started_at=as_of,
                    status="running",
                    attempt=_ATTEMPT,
                    heartbeat_interval_s=self._config.orchestrator.heartbeat.interval_s,
                )
            )
        except Exception as exc:
            _logger.error("pipeline_db_unavailable_at_startup", error=redact_text(repr(exc)))
            raise PipelineStartupError("database unavailable at pipeline startup") from exc

        ctx = AgentContext(
            run_id=run_id,
            as_of=as_of,
            config=self._config,
            db=self._db,
            clock=FrozenClock(as_of),  # pinned — every adapter's now() == as_of
            logger=_logger,
            heartbeat=self._make_heartbeat(run_id),
        )
        ctx.heartbeat()  # initial beat right after the row exists
        result = self._agent.run(ctx)

        status = self._map_status(result)
        # §L8: a failure on this terminal write PROPAGATES — we never swallow it
        # or fabricate a status. The dangling 'running' row is reaped by the next
        # run's orphan sweep.
        self._runs.update_status(
            run_id=run_id,
            agent=self._agent.name,
            attempt=_ATTEMPT,
            status=status,
            finished_at=self._real_clock.now(),
            metrics=dict(result.metrics),
            error=self._serialize_errors(result.errors),
        )
        _logger.info("pipeline_run_complete", run_id=str(run_id), status=status)
        # v1 seam: the downstream Research/Modeling/Briefing agents don't exist
        # yet. When they do, gate each before invoking with
        #   agent.lineage.check_preconditions(
        #       run_id=str(run_id),
        #       prior_statuses=self._runs.prior_statuses(run_id=run_id))
        return status

    # -- internals ----------------------------------------------------------
    @contextmanager
    def _singleton_lock(self) -> Iterator[bool]:
        """Hold the cluster-wide singleton advisory lock for the whole run so
        two overlapping invocations cannot run concurrently (and thus cannot
        orphan-sweep each other's live rows). A DB failure while acquiring the
        lock is a startup failure — no row exists yet, so it surfaces out-of-
        band as `PipelineStartupError`."""
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

    def _make_heartbeat(self, run_id: UUID) -> Callable[[], None]:
        """Build the per-run emitter. Closes over the REAL clock so beats reflect
        wall time even though the agent only sees the pinned clock. Best-effort:
        a heartbeat hiccup must never abort a long ingest."""

        def _emit() -> None:
            try:
                self._runs.heartbeat(
                    run_id=run_id,
                    agent=self._agent.name,
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
