"""Fixtures for MonitorAgent tests (§Q)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tennis.core.config import AppConfig, load_config


@pytest.fixture(scope="session")
def base_config() -> AppConfig:
    root = Path(__file__).resolve().parents[4]
    return load_config(root / "config" / "config.yaml")
