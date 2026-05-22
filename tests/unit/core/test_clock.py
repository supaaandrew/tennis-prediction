"""Clock tests — UTC enforcement, advance semantics, Protocol conformance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from tennis.core.clock import Clock, FrozenClock, RealClock


class TestRealClock:
    def test_now_is_utc(self) -> None:
        clock = RealClock()
        now = clock.now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_implements_protocol(self) -> None:
        clock = RealClock()
        assert isinstance(clock, Clock)


class TestFrozenClock:
    def test_returns_constructed_instant(self) -> None:
        t = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
        clock = FrozenClock(t)
        assert clock.now() == t

    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            FrozenClock(datetime(2026, 5, 21, 12, 0, 0))

    def test_converts_non_utc_to_utc(self) -> None:
        pst = timezone(timedelta(hours=-8))
        t = datetime(2026, 5, 21, 4, 0, 0, tzinfo=pst)
        clock = FrozenClock(t)
        assert clock.now().utcoffset() == timedelta(0)
        assert clock.now().hour == 12  # 04:00 PST == 12:00 UTC

    def test_advance_moves_forward(self) -> None:
        t = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
        clock = FrozenClock(t)
        clock.advance(timedelta(hours=3))
        assert clock.now() == datetime(2026, 5, 21, 15, 0, 0, tzinfo=UTC)

    def test_advance_accepts_negative(self) -> None:
        t = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
        clock = FrozenClock(t)
        clock.advance(timedelta(hours=-1))
        assert clock.now() == datetime(2026, 5, 21, 11, 0, 0, tzinfo=UTC)

    def test_set_replaces_instant(self) -> None:
        clock = FrozenClock(datetime(2020, 1, 1, tzinfo=UTC))
        clock.set(datetime(2026, 5, 21, tzinfo=UTC))
        assert clock.now().year == 2026

    def test_set_rejects_naive(self) -> None:
        clock = FrozenClock(datetime(2026, 5, 21, tzinfo=UTC))
        with pytest.raises(ValueError, match="timezone-aware"):
            clock.set(datetime(2026, 1, 1))

    def test_implements_protocol(self, frozen_clock: FrozenClock) -> None:
        assert isinstance(frozen_clock, Clock)
