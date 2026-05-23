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
