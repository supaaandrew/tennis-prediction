"""Contract dataclass invariants.

Currently only AgentContext is invariant-tested — its `as_of` field must be
timezone-aware to mirror the UTC-only contract on Clock/FrozenClock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from tennis.core.clock import FrozenClock
from tennis.core.contracts import AgentContext


class TestAgentContextAsOf:
    def _ctx(self, as_of: datetime) -> AgentContext:
        return AgentContext(
            run_id=uuid4(),
            as_of=as_of,
            config=None,
            db=None,
            clock=FrozenClock(datetime(2026, 5, 21, tzinfo=UTC)),
            logger=None,
        )

    def test_tz_aware_accepted(self) -> None:
        ctx = self._ctx(datetime(2026, 5, 21, 12, 0, tzinfo=UTC))
        assert ctx.as_of.tzinfo is not None

    def test_naive_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            self._ctx(datetime(2026, 5, 21, 12, 0))


class TestAgentContextHeartbeat:
    """T20 — the per-run heartbeat field is optional with a no-op default, so
    every pre-existing AgentContext construction keeps working (backward compat)
    and agents run outside a pipeline get a safe no-op (§L7)."""

    def _ctx(self, **overrides: object) -> AgentContext:
        kwargs: dict[str, object] = dict(
            run_id=uuid4(),
            as_of=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
            config=None,
            db=None,
            clock=FrozenClock(datetime(2026, 5, 21, tzinfo=UTC)),
            logger=None,
        )
        kwargs.update(overrides)
        return AgentContext(**kwargs)  # type: ignore[arg-type]

    def test_default_is_callable_noop(self) -> None:
        ctx = self._ctx()  # no heartbeat supplied — mirrors existing call sites
        assert callable(ctx.heartbeat)
        assert ctx.heartbeat() is None  # called with no self/args — slot attr

    def test_override_is_invoked(self) -> None:
        calls: list[int] = []
        ctx = self._ctx(heartbeat=lambda: calls.append(1))
        ctx.heartbeat()
        ctx.heartbeat()
        assert calls == [1, 1]
