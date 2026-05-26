"""Shared fixtures for the Modeling Agent ML-internals tests (M1a).

Fast tests (feature_set / assembly / splits / noise / artifacts) avoid the
heavy xgboost/lightgbm import; only base_learners + agent tests pull it in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tennis.core.config import AppConfig, load_config
from tennis.storage.postgres.rows import FeatureSpecRow
from tennis.agents.research.specs import _REGISTRY


@pytest.fixture(scope="session")
def real_config() -> AppConfig:
    root = Path(__file__).resolve().parents[3]
    return load_config(root / "config" / "config.yaml")


@pytest.fixture
def active_specs() -> list[FeatureSpecRow]:
    """Every active feature spec (all 9 families flattened) — the catalog the
    Modeling feature-set resolver reads via `FeatureSpecRepository.list_active`."""
    return [row for rows in _REGISTRY.values() for row in rows]
