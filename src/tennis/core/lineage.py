"""Pipeline run lineage: heartbeat policy, orphan detection, agent preconditions.

This module is pure: no DB access, no global clock, no IO. Everything the
caller needs to fix or query is passed in. The repository layer (in
storage/postgres) wires these primitives to actual `pipeline_runs` rows.

The two primitives are:

`Precondition`
    Declares "agent X must have reached status Y in this run before I'm
    allowed to start." Agents declare a tuple of these as a class attribute;
    the orchestrator (or `Precondition.check`) verifies before invoking
    `Agent.run`. Failure raises `PreconditionNotMetError`, which is loud and
    typed — never silent, never recoverable inside the agent.

`HeartbeatPolicy`
    The (interval, orphan_after) pair the orchestrator promises. Both are
    persisted on `pipeline_runs` (interval as a column; orphan_after via
    the orchestrator's config). `is_orphan` uses `started_at` as the
    reference if no heartbeat has been written yet — so a run that crashed
    before its first heartbeat is still detectable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from tennis.core.errors import PreconditionNotMetError

RunStatus = Literal["running", "succeeded", "failed", "partial"]

_TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "partial"})


# ---------------------------------------------------------------------------
# Heartbeat policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HeartbeatPolicy:
    """How often the orchestrator updates `last_heartbeat_at`, and when a
    `running` row should be considered dead.

    Defaults match the locked decision: 30s interval, 5min orphan threshold.
    """

    interval_s: int = 30
    orphan_after_s: int = 300

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {self.interval_s}")
        if self.orphan_after_s <= self.interval_s:
            raise ValueError(
                f"orphan_after_s ({self.orphan_after_s}) must exceed "
                f"interval_s ({self.interval_s}) — give the orchestrator at "
                f"least one cadence to recover before being declared dead."
            )

    def is_orphan(
        self,
        *,
        status: str,
        started_at: datetime,
        last_heartbeat_at: datetime | None,
        now: datetime,
    ) -> bool:
        """True iff a `running` row has gone silent for longer than
        `orphan_after_s`. Uses `started_at` as the reference instant when
        no heartbeat has been written yet (the run crashed pre-first-beat).
        """
        if status != "running":
            return False
        reference = last_heartbeat_at if last_heartbeat_at is not None else started_at
        # Guard against naive datetimes — if mixed-tz arithmetic blows up,
        # better to fail loudly than silently classify as orphaned/not.
        if reference.tzinfo is None or now.tzinfo is None:
            raise ValueError("is_orphan requires timezone-aware datetimes")
        return (now - reference) > timedelta(seconds=self.orphan_after_s)


# ---------------------------------------------------------------------------
# Agent precondition
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Precondition:
    """Declares that an agent requires a prior agent to have reached a
    given status within the same run_id.

    Usage (on an Agent class):
        preconditions: tuple[Precondition, ...] = (
            Precondition(previous_agent="data"),
        )

    The orchestrator (or any caller) invokes `check(...)` with a mapping of
    agent_name → terminal status as recorded in pipeline_runs for this run.
    """

    previous_agent: str
    required_status: RunStatus = "succeeded"

    def check(
        self,
        *,
        run_id: str,
        prior_statuses: Mapping[str, str],
    ) -> None:
        """Raise `PreconditionNotMetError` if the prior agent did not reach
        `required_status`. `prior_statuses` should ONLY contain terminal
        statuses (we never gate on a still-`running` prior agent).
        """
        actual = prior_statuses.get(self.previous_agent)
        if actual is None or actual != self.required_status:
            raise PreconditionNotMetError(
                agent=self.previous_agent,
                expected=self.required_status,
                actual=actual,
                run_id=run_id,
            )


def all_terminal(statuses: Mapping[str, str]) -> bool:
    """Helper: True iff every value in `statuses` is a terminal status."""
    return all(s in _TERMINAL_STATUSES for s in statuses.values())


# ---------------------------------------------------------------------------
# Agent declaration helper — exposed so concrete Agent classes can carry a
# tuple of Precondition without re-importing from contracts.py.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AgentLineage:
    """Bundle of lineage knobs an Agent declares as a class attribute."""

    preconditions: tuple[Precondition, ...] = field(default_factory=tuple)
    heartbeat: HeartbeatPolicy = field(default_factory=HeartbeatPolicy)

    def check_preconditions(
        self,
        *,
        run_id: str,
        prior_statuses: Mapping[str, str],
    ) -> None:
        for p in self.preconditions:
            p.check(run_id=run_id, prior_statuses=prior_statuses)
