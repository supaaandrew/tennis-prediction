"""Pipeline run lineage tests — heartbeat policy + precondition checks.

TDD-required per Day 2 instructions. Every branch of HeartbeatPolicy.is_orphan
and Precondition.check gets its own test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tennis.core.errors import PreconditionNotMetError
from tennis.core.lineage import (
    AgentLineage,
    HeartbeatPolicy,
    Precondition,
    all_terminal,
)


# ---------------------------------------------------------------------------
# HeartbeatPolicy
# ---------------------------------------------------------------------------
class TestHeartbeatPolicyConstruction:
    def test_defaults_match_locked_decision(self) -> None:
        p = HeartbeatPolicy()
        assert p.interval_s == 30
        assert p.orphan_after_s == 300

    def test_zero_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="interval_s"):
            HeartbeatPolicy(interval_s=0, orphan_after_s=300)

    def test_negative_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="interval_s"):
            HeartbeatPolicy(interval_s=-1, orphan_after_s=300)

    def test_orphan_threshold_must_exceed_interval(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            HeartbeatPolicy(interval_s=30, orphan_after_s=30)
        with pytest.raises(ValueError, match="must exceed"):
            HeartbeatPolicy(interval_s=30, orphan_after_s=10)


class TestHeartbeatPolicyIsOrphan:
    @pytest.fixture
    def policy(self) -> HeartbeatPolicy:
        return HeartbeatPolicy(interval_s=30, orphan_after_s=300)

    @pytest.fixture
    def now(self) -> datetime:
        return datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)

    def test_succeeded_is_never_orphan(self, policy: HeartbeatPolicy, now: datetime) -> None:
        assert (
            policy.is_orphan(
                status="succeeded",
                started_at=now - timedelta(hours=1),
                last_heartbeat_at=None,
                now=now,
            )
            is False
        )

    def test_failed_is_never_orphan(self, policy: HeartbeatPolicy, now: datetime) -> None:
        assert (
            policy.is_orphan(
                status="failed",
                started_at=now - timedelta(hours=1),
                last_heartbeat_at=None,
                now=now,
            )
            is False
        )

    def test_running_with_fresh_heartbeat_is_not_orphan(
        self, policy: HeartbeatPolicy, now: datetime
    ) -> None:
        assert (
            policy.is_orphan(
                status="running",
                started_at=now - timedelta(minutes=10),
                last_heartbeat_at=now - timedelta(seconds=20),
                now=now,
            )
            is False
        )

    def test_running_with_stale_heartbeat_is_orphan(
        self, policy: HeartbeatPolicy, now: datetime
    ) -> None:
        assert (
            policy.is_orphan(
                status="running",
                started_at=now - timedelta(minutes=10),
                last_heartbeat_at=now - timedelta(seconds=301),
                now=now,
            )
            is True
        )

    def test_running_no_heartbeat_recent_start_is_not_orphan(
        self, policy: HeartbeatPolicy, now: datetime
    ) -> None:
        # Started 10 seconds ago, no heartbeat yet → not orphan
        assert (
            policy.is_orphan(
                status="running",
                started_at=now - timedelta(seconds=10),
                last_heartbeat_at=None,
                now=now,
            )
            is False
        )

    def test_running_no_heartbeat_ancient_start_is_orphan(
        self, policy: HeartbeatPolicy, now: datetime
    ) -> None:
        # Started 10 minutes ago, never wrote a heartbeat → orphan
        assert (
            policy.is_orphan(
                status="running",
                started_at=now - timedelta(minutes=10),
                last_heartbeat_at=None,
                now=now,
            )
            is True
        )

    def test_naive_datetime_raises(self, policy: HeartbeatPolicy, now: datetime) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            policy.is_orphan(
                status="running",
                started_at=datetime(2026, 5, 21),  # naive
                last_heartbeat_at=None,
                now=now,
            )


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------
class TestPrecondition:
    def test_passes_when_required_status_matches(self) -> None:
        Precondition(previous_agent="data").check(
            run_id="r1",
            prior_statuses={"data": "succeeded"},
        )  # no exception

    def test_raises_when_previous_agent_absent(self) -> None:
        with pytest.raises(PreconditionNotMetError) as excinfo:
            Precondition(previous_agent="data").check(
                run_id="r1",
                prior_statuses={},
            )
        err = excinfo.value
        assert err.agent == "data"
        assert err.expected == "succeeded"
        assert err.actual is None
        assert err.run_id == "r1"

    def test_raises_when_previous_agent_failed(self) -> None:
        with pytest.raises(PreconditionNotMetError) as excinfo:
            Precondition(previous_agent="data").check(
                run_id="r1",
                prior_statuses={"data": "failed"},
            )
        assert excinfo.value.actual == "failed"

    def test_raises_when_previous_agent_partial(self) -> None:
        # "partial" is a terminal status but not "succeeded" — must still raise.
        with pytest.raises(PreconditionNotMetError):
            Precondition(previous_agent="data").check(
                run_id="r1",
                prior_statuses={"data": "partial"},
            )

    def test_custom_required_status(self) -> None:
        # If the orchestrator allows "partial" as upstream OK, the
        # precondition can be tuned per agent.
        Precondition(previous_agent="data", required_status="partial").check(
            run_id="r1",
            prior_statuses={"data": "partial"},
        )


class TestAgentLineageBundle:
    def test_no_preconditions_passes_trivially(self) -> None:
        AgentLineage().check_preconditions(run_id="r1", prior_statuses={})

    def test_all_preconditions_enforced(self) -> None:
        lineage = AgentLineage(
            preconditions=(
                Precondition(previous_agent="data"),
                Precondition(previous_agent="research"),
            )
        )
        with pytest.raises(PreconditionNotMetError) as excinfo:
            lineage.check_preconditions(
                run_id="r1",
                prior_statuses={"data": "succeeded"},  # research missing
            )
        assert excinfo.value.agent == "research"

    def test_heartbeat_policy_defaults(self) -> None:
        assert AgentLineage().heartbeat == HeartbeatPolicy()


class TestAllTerminal:
    def test_empty_is_terminal(self) -> None:
        assert all_terminal({}) is True

    def test_running_is_not_terminal(self) -> None:
        assert all_terminal({"a": "running"}) is False

    def test_mixed_terminals_ok(self) -> None:
        assert (
            all_terminal({"a": "succeeded", "b": "failed", "c": "partial"})
            is True
        )

    def test_one_running_among_terminals_fails(self) -> None:
        assert (
            all_terminal({"a": "succeeded", "b": "running"})
            is False
        )
