"""Injectable clock abstraction.

Every part of the system that asks "what time is it?" goes through `Clock`.
This makes feature-engineering tests deterministic and makes it possible to
freeze the clock at any historical instant without touching production code.

All clocks return timezone-aware UTC datetimes — naive datetimes are rejected
at construction time and never produced.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """The only contract anything in the project should depend on for time."""

    def now(self) -> datetime:  # pragma: no cover - protocol
        """Return the current instant as a timezone-aware UTC datetime."""
        ...


class RealClock:
    """Production clock. Reads the OS clock; always returns UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Test / replay clock. Returns a fixed instant until `advance` is called.

    Construction rejects naive datetimes — every Clock in the system is UTC.
    """

    def __init__(self, frozen_at: datetime) -> None:
        if frozen_at.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime")
        self._t: datetime = frozen_at.astimezone(UTC)

    def now(self) -> datetime:
        return self._t

    def advance(self, delta: timedelta) -> None:
        """Move the frozen instant forward by `delta`. Negative delta allowed."""
        self._t = self._t + delta

    def set(self, instant: datetime) -> None:
        """Reset the clock to an arbitrary instant. Must be tz-aware."""
        if instant.tzinfo is None:
            raise ValueError("FrozenClock.set requires a timezone-aware datetime")
        self._t = instant.astimezone(UTC)
