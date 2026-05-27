"""DailyPipeline lifecycle tests (P7).

Exercises real control flow — orphan sweep, status mapping, the DB-startup guard,
the §L8 update-status propagation, and the two-clock split — with an in-memory
fake `PipelineRunRepository` (no Docker) and a programmable fake agent.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from tennis.agents.orchestrator import DailyPipeline
from tennis.agents.orchestrator.pipeline import _DAILY_LOCK_KEY
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import PipelineStartupError, PreconditionNotMetError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy, Precondition
from tennis.storage.postgres.rows import PipelineRunRow

_NOW = datetime(2026, 5, 24, 6, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[4]
    return load_config(root / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# In-memory PipelineRunRepository — faithful to the Protocol, no DB.
# ---------------------------------------------------------------------------
class _FakeRuns:
    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, str, int], PipelineRunRow] = {}
        self.beats: list[datetime] = []
        self.fail_insert = False
        self.fail_update = False
        self.lock_available = True  # set False to simulate a concurrent run
        self.fail_lock = False  # set True to simulate a DB failure ACQUIRING the lock
        self.lock_keys: list[int] = []
        # When set, prior_statuses returns this verbatim — lets a precondition-gate
        # test simulate a prior agent's terminal status under the fresh run_id
        # (which is otherwise empty; see the §R6 shared-run_id caveat).
        self.force_prior: dict[str, str] | None = None

    @staticmethod
    def _key(run_id: UUID, agent: str, attempt: int) -> tuple[UUID, str, int]:
        return (run_id, agent, attempt)

    def insert(self, row: PipelineRunRow) -> PipelineRunRow:
        if self.fail_insert:
            raise RuntimeError("connection refused")
        self.rows[self._key(row.run_id, row.agent, row.attempt)] = row
        return row

    def get(self, *, run_id: UUID, agent: str, attempt: int = 1) -> PipelineRunRow | None:
        return self.rows.get(self._key(run_id, agent, attempt))

    def update_status(
        self, *, run_id: UUID, agent: str, attempt: int, status: str,
        finished_at: datetime | None = None,
        metrics: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if self.fail_update:
            raise RuntimeError("terminal write failed")
        key = self._key(run_id, agent, attempt)
        self.rows[key] = dataclasses.replace(
            self.rows[key], status=status, finished_at=finished_at,
            metrics=metrics if metrics is not None else self.rows[key].metrics,
            error=error,
        )

    def heartbeat(self, *, run_id: UUID, agent: str, attempt: int, now: datetime) -> None:
        self.beats.append(now)
        key = self._key(run_id, agent, attempt)
        if key in self.rows:
            self.rows[key] = dataclasses.replace(self.rows[key], last_heartbeat_at=now)

    def orphans(self, *, orphan_after_s: int, now: datetime) -> list[PipelineRunRow]:
        threshold = now - timedelta(seconds=orphan_after_s)
        out = []
        for row in self.rows.values():
            if row.status != "running":
                continue
            ref = row.last_heartbeat_at or row.started_at
            if ref < threshold:
                out.append(row)
        return out

    def mark_failed_with_reason(
        self, *, run_id: UUID, agent: str, attempt: int, reason: str
    ) -> None:
        key = self._key(run_id, agent, attempt)
        self.rows[key] = dataclasses.replace(
            self.rows[key], status="failed", error={"reason": reason}
        )

    def prior_statuses(self, *, run_id: UUID) -> dict[str, str]:
        if self.force_prior is not None:
            return dict(self.force_prior)
        return {
            agent: row.status
            for (rid, agent, _attempt), row in self.rows.items()
            if rid == run_id
        }

    @contextmanager
    def advisory_lock(self, *, key: int) -> Iterator[bool]:
        self.lock_keys.append(key)
        if self.fail_lock:  # DB failure while acquiring the lock (in __enter__)
            raise RuntimeError("lock acquisition failed: connection refused")
        yield self.lock_available


class _FakeAgent:
    name = "data"

    def __init__(
        self, result: AgentResult, *, on_run: Any = None
    ) -> None:
        self.lineage = AgentLineage(preconditions=(), heartbeat=HeartbeatPolicy())
        self._result = result
        self._on_run = on_run
        self.ctx: AgentContext | None = None

    def run(self, ctx: AgentContext) -> AgentResult:
        self.ctx = ctx
        if self._on_run is not None:
            self._on_run(ctx)
        return self._result


def _pipeline(
    config: AppConfig, runs: _FakeRuns, agent: _FakeAgent, *, clock: FrozenClock | None = None
) -> DailyPipeline:
    # Single-agent convenience: wrap as a 1-element chain. `run_once()`'s
    # aggregate over one agent equals that agent's terminal status.
    return DailyPipeline(
        real_clock=clock or FrozenClock(_NOW),
        runs=runs,  # type: ignore[arg-type]
        agents=[agent],  # type: ignore[list-item]
        config=config,
        db=None,
    )


def _chain(
    config: AppConfig,
    runs: _FakeRuns,
    agents: list[Any],
    *,
    clock: FrozenClock | None = None,
) -> DailyPipeline:
    return DailyPipeline(
        real_clock=clock or FrozenClock(_NOW),
        runs=runs,  # type: ignore[arg-type]
        agents=agents,  # type: ignore[arg-type]
        config=config,
        db=None,
    )


def _ok_result() -> AgentResult:
    return AgentResult(ok=True, metrics={"sackmann": {"complete": True}}, errors=())


class _ChainAgent:
    """Configurable fake agent for multi-agent chain tests: arbitrary name,
    declared preconditions, terminal result, and an optional run() exception."""

    def __init__(
        self,
        name: str,
        result: AgentResult | None = None,
        *,
        preconditions: tuple[Precondition, ...] = (),
        raise_exc: Exception | None = None,
    ) -> None:
        self.name = name
        self.lineage = AgentLineage(
            preconditions=preconditions, heartbeat=HeartbeatPolicy()
        )
        self._result = result if result is not None else _ok_result()
        self._raise = raise_exc
        self.ctx: AgentContext | None = None

    def run(self, ctx: AgentContext) -> AgentResult:
        self.ctx = ctx
        if self._raise is not None:
            raise self._raise
        return self._result


def _partial_result() -> AgentResult:
    return AgentResult(
        ok=False, metrics={}, errors=(AgentError(code="odds_error", message="x"),)
    )


def _failed_result() -> AgentResult:
    return AgentResult(
        ok=False, metrics={}, errors=(AgentError(code="staleness_halt", message="stale"),)
    )


# ---------------------------------------------------------------------------
# T11 — orphan sweep
# ---------------------------------------------------------------------------
class TestOrphanSweep:
    def test_stale_running_row_marked_failed_then_new_run_proceeds(
        self, config: AppConfig
    ) -> None:
        runs = _FakeRuns()
        stale_id = uuid4()
        runs.insert(PipelineRunRow(
            run_id=stale_id, pipeline="daily", agent="data",
            started_at=_NOW - timedelta(seconds=600), status="running",
            attempt=1, last_heartbeat_at=None, heartbeat_interval_s=30,
        ))
        agent = _FakeAgent(_ok_result())

        status = _pipeline(config, runs, agent).run_once()

        # stale row reaped
        stale = runs.get(run_id=stale_id, agent="data")
        assert stale is not None and stale.status == "failed"
        assert stale.error == {"reason": "orphaned: no heartbeat within orphan_after_s"}
        # new run completed
        assert status == "succeeded"


# ---------------------------------------------------------------------------
# T12 / T13 / T14 — terminal status persisted
# ---------------------------------------------------------------------------
class TestStatusPersisted:
    def test_succeeded_persists_finished_at_and_metrics(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        agent = _FakeAgent(_ok_result())

        status = _pipeline(config, runs, agent).run_once()

        assert status == "succeeded"
        row = runs.get(run_id=agent.ctx.run_id, agent="data")  # type: ignore[union-attr]
        assert row is not None
        assert row.status == "succeeded"
        assert row.finished_at is not None
        assert row.metrics == {"sackmann": {"complete": True}}
        assert runs.beats  # at least the initial heartbeat fired

    def test_partial_persists_serialized_errors(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        agent = _FakeAgent(AgentResult(
            ok=False, metrics={"odds": {"complete": False}},
            errors=(AgentError(code="odds_error", message="boom", cause="RuntimeError()"),),
        ))

        status = _pipeline(config, runs, agent).run_once()

        assert status == "partial"
        row = runs.get(run_id=agent.ctx.run_id, agent="data")  # type: ignore[union-attr]
        assert row is not None and row.status == "partial"
        assert row.error == {
            "errors": [{"code": "odds_error", "message": "boom", "cause": "RuntimeError()"}]
        }

    def test_staleness_halt_ends_failed_not_running(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        agent = _FakeAgent(AgentResult(
            ok=False, metrics={},
            errors=(AgentError(code="staleness_halt", message="stale"),),
        ))

        status = _pipeline(config, runs, agent).run_once()

        assert status == "failed"
        row = runs.get(run_id=agent.ctx.run_id, agent="data")  # type: ignore[union-attr]
        assert row is not None and row.status == "failed"  # not left 'running'

    def test_no_active_model_ends_failed(self, config: AppConfig) -> None:
        # M1b prediction mode: no usable model → 'failed' (in _FATAL_CODES).
        runs = _FakeRuns()
        agent = _FakeAgent(AgentResult(
            ok=False, metrics={},
            errors=(AgentError(code="no_active_model", message="none"),),
        ))
        assert _pipeline(config, runs, agent).run_once() == "failed"

    def test_calibration_degraded_ends_partial(self, config: AppConfig) -> None:
        # M1b: degraded calibration → 'partial' (NOT in _FATAL_CODES) — the model
        # is still served uncalibrated.
        runs = _FakeRuns()
        agent = _FakeAgent(AgentResult(
            ok=False, metrics={},
            errors=(AgentError(code="calibration_degraded", message="degraded"),),
        ))
        assert _pipeline(config, runs, agent).run_once() == "partial"


class TestFatalCodes:
    def test_membership(self) -> None:
        # pre-step 6.1: no_active_model is fatal; calibration_degraded is not.
        from tennis.agents.orchestrator.pipeline import _FATAL_CODES

        assert "no_active_model" in _FATAL_CODES
        assert "calibration_degraded" not in _FATAL_CODES


# ---------------------------------------------------------------------------
# T15 — DB unavailable at startup
# ---------------------------------------------------------------------------
class TestDbUnavailableAtStartup:
    def test_insert_failure_raises_startup_error_no_row(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.orchestrator.pipeline._logger", mock_logger)
        runs = _FakeRuns()
        runs.fail_insert = True
        agent = _FakeAgent(_ok_result())

        with pytest.raises(PipelineStartupError):
            _pipeline(config, runs, agent).run_once()

        assert runs.rows == {}  # no dangling row
        assert agent.ctx is None  # agent never ran
        mock_logger.error.assert_any_call(
            "pipeline_db_unavailable_at_startup", error="RuntimeError('connection refused')"
        )


# ---------------------------------------------------------------------------
# T16 — terminal update_status failure propagates (§L8)
# ---------------------------------------------------------------------------
class TestUpdateStatusFailsGracefully:
    def test_update_failure_propagates_row_stays_running(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        runs.fail_update = True
        agent = _FakeAgent(_ok_result())

        with pytest.raises(RuntimeError, match="terminal write failed"):
            _pipeline(config, runs, agent).run_once()

        # agent ran, but the terminal write failed — row is left 'running' for
        # the next day's orphan sweep (§L8). No swallow, no fabricated status.
        row = runs.get(run_id=agent.ctx.run_id, agent="data")  # type: ignore[union-attr]
        assert row is not None and row.status == "running"


# ---------------------------------------------------------------------------
# T17 — precondition input + clock split
# ---------------------------------------------------------------------------
class TestPreconditionsAndClock:
    def test_prior_statuses_reports_data_status(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        agent = _FakeAgent(_ok_result())

        _pipeline(config, runs, agent).run_once()

        run_id = agent.ctx.run_id  # type: ignore[union-attr]
        # feeds the future downstream precondition gate
        assert runs.prior_statuses(run_id=run_id) == {"data": "succeeded"}

    def test_agent_sees_pinned_clock_while_heartbeat_uses_real_clock(
        self, config: AppConfig
    ) -> None:
        runs = _FakeRuns()
        real_clock = FrozenClock(_NOW)

        def _advance_and_beat(ctx: AgentContext) -> None:
            real_clock.advance(timedelta(minutes=5))  # wall time moves on
            ctx.heartbeat()

        agent = _FakeAgent(_ok_result(), on_run=_advance_and_beat)
        _pipeline(config, runs, agent, clock=real_clock).run_once()

        ctx = agent.ctx
        assert ctx is not None
        # the agent's clock is pinned to as_of (run start), never advances
        assert ctx.clock.now() == _NOW
        assert ctx.as_of == _NOW
        # heartbeats read the REAL clock: initial beat at _NOW, in-run beat later
        assert runs.beats[0] == _NOW
        assert runs.beats[-1] == _NOW + timedelta(minutes=5)


# ---------------------------------------------------------------------------
# Singleton run lock (Codex HIGH fix) — a concurrent run is rejected cleanly
# ---------------------------------------------------------------------------
class TestSingletonLock:
    def test_concurrent_run_rejected_without_inserting_a_row(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.orchestrator.pipeline._logger", mock_logger)
        runs = _FakeRuns()
        runs.lock_available = False  # another run already holds the lock
        agent = _FakeAgent(_ok_result())

        status = _pipeline(config, runs, agent).run_once()

        # clean no-op: no status, no run row, orphan sweep never ran, agent
        # never invoked — so it cannot reap the live sibling run's row (§L7).
        assert status is None
        assert runs.rows == {}
        assert runs.beats == []
        assert agent.ctx is None
        assert runs.lock_keys == [_DAILY_LOCK_KEY]
        mock_logger.warning.assert_any_call(
            "pipeline_already_running", lock_key=_DAILY_LOCK_KEY
        )

    def test_lock_acquired_and_released_for_a_normal_run(self, config: AppConfig) -> None:
        runs = _FakeRuns()  # lock_available=True
        agent = _FakeAgent(_ok_result())

        status = _pipeline(config, runs, agent).run_once()

        # lock was taken once for the run, and the run proceeded normally
        assert status == "succeeded"
        assert runs.lock_keys == [_DAILY_LOCK_KEY]
        assert agent.ctx is not None

    def test_lock_acquisition_db_failure_normalized_to_startup_error(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A DB failure while ACQUIRING the lock (in __enter__) is a startup
        # failure: normalized to PipelineStartupError (§L2), no row, agent never
        # ran. (Regression for the §S restored typed-error contract.)
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.orchestrator.pipeline._logger", mock_logger)
        runs = _FakeRuns()
        runs.fail_lock = True
        agent = _FakeAgent(_ok_result())

        with pytest.raises(PipelineStartupError):
            _pipeline(config, runs, agent).run_once()

        assert runs.rows == {}  # no dangling row
        assert agent.ctx is None  # agent never ran
        mock_logger.error.assert_any_call(
            "pipeline_db_unavailable_at_startup",
            error="RuntimeError('lock acquisition failed: connection refused')",
        )


# ---------------------------------------------------------------------------
# R6a — precondition gate placed before agent.run() (M-e)
# ---------------------------------------------------------------------------
class _GatedAgent:
    """An agent declaring a `data`-succeeded precondition (like ResearchAgent),
    so the pipeline's gate is actually exercised."""

    name = "research"

    def __init__(self, result: AgentResult) -> None:
        self.lineage = AgentLineage(
            preconditions=(
                Precondition(previous_agent="data", required_status="succeeded"),
            ),
            heartbeat=HeartbeatPolicy(),
        )
        self._result = result
        self.ctx: AgentContext | None = None

    def run(self, ctx: AgentContext) -> AgentResult:
        self.ctx = ctx
        return self._result


class TestPreconditionGate:
    def test_empty_preconditions_is_a_noop_data_agent_runs(self, config: AppConfig) -> None:
        # DataAgent has no preconditions, so the gate never blocks it.
        runs = _FakeRuns()
        agent = _FakeAgent(_ok_result())
        status = _pipeline(config, runs, agent).run_once()
        assert status == "succeeded"
        assert agent.ctx is not None  # agent ran

    def test_gate_not_met_skips_downstream_no_row_no_raise(
        self, config: AppConfig
    ) -> None:
        # data runs and is PARTIAL → research's precondition (data succeeded) is
        # not met → research is SKIPPED in chain mode: it writes NO row, raises
        # nothing, and the loop simply moves on (§S4).
        runs = _FakeRuns()
        data = _ChainAgent("data", _partial_result())
        research = _ChainAgent(
            "research",
            preconditions=(
                Precondition(previous_agent="data", required_status="succeeded"),
            ),
        )
        status = _chain(config, runs, [data, research]).run_once()
        run_id = data.ctx.run_id  # type: ignore[union-attr]
        assert research.ctx is None  # skipped — never ran
        assert runs.get(run_id=run_id, agent="research") is None  # no row
        assert runs.get(run_id=run_id, agent="data").status == "partial"  # type: ignore[union-attr]
        # aggregate over the only agent that ran (data) → partial. No raise.
        assert status == "partial"

    def test_gate_passes_when_data_succeeded(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        runs.force_prior = {"data": "succeeded"}  # simulate the prior agent
        agent = _GatedAgent(_ok_result())
        status = _pipeline(config, runs, agent).run_once()
        assert status == "succeeded"
        assert agent.ctx is not None  # gate passed → agent ran

    def test_agent_exception_records_failed_and_continues_to_monitor(
        self, config: AppConfig
    ) -> None:
        # An exception escaping run() (a bug, the §M9 IdempotencyError, etc.)
        # terminates THIS agent's row as 'failed' and the loop CONTINUES so the
        # Monitor still runs (§S4) — it is NOT re-raised.
        runs = _FakeRuns()
        boom = _ChainAgent("research", raise_exc=RuntimeError("boom mid-run"))
        monitor = _ChainAgent("monitor")  # empty preconditions → always runs
        status = _chain(config, runs, [boom, monitor]).run_once()  # must NOT raise
        run_id = boom.ctx.run_id  # type: ignore[union-attr]
        boom_row = runs.get(run_id=run_id, agent="research")
        assert boom_row is not None and boom_row.status == "failed"
        assert "RuntimeError" in boom_row.error["exception"]  # type: ignore[index]
        # Monitor STILL ran despite the upstream crash (A13 / §S4).
        assert monitor.ctx is not None
        mon_row = runs.get(run_id=run_id, agent="monitor")
        assert mon_row is not None and mon_row.status == "succeeded"
        assert status == "failed"  # aggregate: the failed agent dominates

    def test_check_preconditions_in_isolation(self) -> None:
        lineage = AgentLineage(
            preconditions=(
                Precondition(previous_agent="data", required_status="succeeded"),
            ),
        )
        # Passes only with the exact required status.
        lineage.check_preconditions(run_id="r", prior_statuses={"data": "succeeded"})
        with pytest.raises(PreconditionNotMetError):
            lineage.check_preconditions(run_id="r", prior_statuses={})
        with pytest.raises(PreconditionNotMetError):
            lineage.check_preconditions(run_id="r", prior_statuses={"data": "partial"})


# ---------------------------------------------------------------------------
# Multi-agent chain under one run_id (§S4)
# ---------------------------------------------------------------------------
class TestMultiAgentChain:
    def _daily_agents(self) -> list[_ChainAgent]:
        return [
            _ChainAgent("data"),
            _ChainAgent(
                "research",
                preconditions=(Precondition("data", "succeeded"),),
            ),
            _ChainAgent(
                "modeling",
                preconditions=(Precondition("research", "succeeded"),),
            ),
            _ChainAgent(
                "briefing",
                preconditions=(Precondition("modeling", "succeeded"),),
            ),
            _ChainAgent("monitor"),  # empty preconditions → always runs (A13)
        ]

    def test_full_chain_runs_all_under_one_run_id(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        agents = self._daily_agents()
        status = _chain(config, runs, agents).run_once()

        # All five stages ran and share ONE run_id.
        run_ids = {a.ctx.run_id for a in agents if a.ctx is not None}
        assert len(run_ids) == 1
        run_id = run_ids.pop()
        for name in ("data", "research", "modeling", "briefing", "monitor"):
            row = runs.get(run_id=run_id, agent=name)
            assert row is not None and row.status == "succeeded"
        # The singleton lock was taken exactly ONCE for the whole chain (§S4).
        assert runs.lock_keys == [_DAILY_LOCK_KEY]
        assert status == "succeeded"

    def test_data_failure_skips_middle_but_monitor_runs(self, config: AppConfig) -> None:
        runs = _FakeRuns()
        agents = self._daily_agents()
        agents[0] = _ChainAgent("data", _failed_result())  # data → failed
        status = _chain(config, runs, agents).run_once()

        run_id = agents[0].ctx.run_id  # type: ignore[union-attr]
        data_row = runs.get(run_id=run_id, agent="data")
        assert data_row is not None and data_row.status == "failed"
        # research / modeling / briefing all skipped: never ran, no rows.
        for name, agent in zip(
            ("research", "modeling", "briefing"), agents[1:4], strict=True
        ):
            assert agent.ctx is None
            assert runs.get(run_id=run_id, agent=name) is None
        # Monitor (empty preconditions) STILL runs despite the upstream failure.
        assert agents[4].ctx is not None
        mon_row = runs.get(run_id=run_id, agent="monitor")
        assert mon_row is not None and mon_row.status == "succeeded"
        # Aggregate: the failed agent dominates → the run is 'failed'.
        assert status == "failed"

    def test_bootstrap_no_active_model_sequence(self, config: AppConfig) -> None:
        """§S6 bootstrap: with no active model, Modeling fails fast
        (`no_active_model`, fatal) → Briefing is skipped (its `modeling
        succeeded` precondition is unmet) → Monitor still runs and reports
        `monitor_no_active_model` (partial, NOT fatal). This is expected, not a
        bug."""
        runs = _FakeRuns()
        no_model = AgentResult(
            ok=False,
            metrics={},
            errors=(AgentError(code="no_active_model", message="none"),),
        )
        monitor_partial = AgentResult(
            ok=False,
            metrics={},
            errors=(
                AgentError(code="monitor_no_active_model", message="no model to score"),
            ),
        )
        agents = [
            _ChainAgent("data"),
            _ChainAgent("research", preconditions=(Precondition("data", "succeeded"),)),
            _ChainAgent(
                "modeling",
                no_model,
                preconditions=(Precondition("research", "succeeded"),),
            ),
            _ChainAgent(
                "briefing", preconditions=(Precondition("modeling", "succeeded"),)
            ),
            _ChainAgent("monitor", monitor_partial),
        ]
        status = _chain(config, runs, agents).run_once()

        run_id = agents[0].ctx.run_id  # type: ignore[union-attr]
        modeling_row = runs.get(run_id=run_id, agent="modeling")
        assert modeling_row is not None and modeling_row.status == "failed"
        # Briefing skipped: precondition (modeling succeeded) unmet → no row.
        assert agents[3].ctx is None
        assert runs.get(run_id=run_id, agent="briefing") is None
        # Monitor still ran and degraded to partial (no model yet is not fatal).
        assert agents[4].ctx is not None
        mon_row = runs.get(run_id=run_id, agent="monitor")
        assert mon_row is not None and mon_row.status == "partial"
        # Aggregate is 'failed' (Modeling) — the daily run did not produce a brief.
        assert status == "failed"
