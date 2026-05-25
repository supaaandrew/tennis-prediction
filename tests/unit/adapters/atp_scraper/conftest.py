"""Fixtures for the ATP scraper unit tests.

HTML is loaded from disk (`tests/fixtures/atp_scraper/`) rather than embedded
as inline strings — page fragments are too bulky and brittle to live in test
files.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "atp_scraper"


@pytest.fixture
def load_fixture() -> Callable[[str], str]:
    def _load(name: str) -> str:
        return (_FIXTURES / name).read_text(encoding="utf-8")

    return _load
