"""Shared pytest fixtures.

Kept minimal for the foundation phase — only utilities needed by core/ tests
live here. Integration / DB fixtures will be added when storage/ ships.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tennis.core.clock import FrozenClock


@pytest.fixture
def fixed_instant() -> datetime:
    return datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def frozen_clock(fixed_instant: datetime) -> FrozenClock:
    return FrozenClock(fixed_instant)


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_path(repo_root: Path) -> Path:
    return repo_root / "config" / "config.yaml"
