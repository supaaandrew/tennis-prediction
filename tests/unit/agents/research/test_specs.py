"""Tests for `specs.py` — the lockstep registry, seeding, and expected_specs builder.

The production `_REGISTRY` is empty in R2 (no extractor exists yet), so the
mechanism is exercised with an injected fixture registry + an in-memory fake
`FeatureSpecRepository`. Covers: idempotent seeding, unknown-family loudness,
critical stamping from `_CRITICAL_FEATURE_KEYS`, and family-scoped expected_specs.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from tennis.agents.research.specs import (
    _REGISTRY,
    build_expected_specs,
    seed_feature_specs,
)
from tennis.agents.research.validator import _CRITICAL_FEATURE_KEYS, FeatureSpec
from tennis.core.errors import FeatureContractError
from tennis.storage.postgres.rows import FeatureSpecRow

# `elo_diff_blended` is in _CRITICAL_FEATURE_KEYS; `alpha_plain` / beta keys are not.
_CRITICAL_KEY = "elo_diff_blended"


class _FakeFeatureSpecRepo:
    """In-memory FeatureSpecRepository: upsert dedups on (feature_key, version)."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, int], FeatureSpecRow] = {}

    def get(self, *, feature_key: str, version: int) -> FeatureSpecRow | None:
        return self._store.get((feature_key, version))

    def list_active(self, *, feature_set: str) -> Sequence[FeatureSpecRow]:
        return tuple(self._store.values())

    def upsert(self, row: FeatureSpecRow) -> FeatureSpecRow:
        self._store[(row.feature_key, row.version)] = row
        return row


@pytest.fixture
def registry() -> dict[str, tuple[FeatureSpecRow, ...]]:
    return {
        "alpha": (
            FeatureSpecRow(_CRITICAL_KEY, 1, "float"),
            FeatureSpecRow("alpha_plain", 1, "int"),
        ),
        "beta": (FeatureSpecRow("beta_feat", 1, "cat"),),
    }


class TestSeedFeatureSpecs:
    def test_upserts_rows_and_returns_count(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()

        n = seed_feature_specs(repo, families=["alpha"], registry=registry)

        assert n == 2
        assert len(repo.list_active(feature_set="v1")) == 2

    def test_is_idempotent(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()

        seed_feature_specs(repo, families=["alpha"], registry=registry)
        seed_feature_specs(repo, families=["alpha"], registry=registry)

        # Re-seeding leaves the catalog unchanged (upsert keyed on key+version).
        assert len(repo.list_active(feature_set="v1")) == 2

    def test_empty_families_is_a_noop(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()

        assert seed_feature_specs(repo, families=[], registry=registry) == 0
        assert len(repo.list_active(feature_set="v1")) == 0

    def test_unknown_family_raises(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()

        with pytest.raises(KeyError):
            seed_feature_specs(repo, families=["does_not_exist"], registry=registry)

    def test_multiple_families_count(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()

        n = seed_feature_specs(repo, families=["alpha", "beta"], registry=registry)

        assert n == 3
        assert len(repo.list_active(feature_set="v1")) == 3


class TestBuildExpectedSpecs:
    def test_stamps_critical_from_the_keyset(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()
        seed_feature_specs(repo, families=["alpha"], registry=registry)

        specs = build_expected_specs(
            repo, feature_set="v1", families=["alpha"], registry=registry
        )

        by_key = {s.feature_key: s for s in specs}
        assert by_key[_CRITICAL_KEY].critical is True
        assert by_key["alpha_plain"].critical is False

    def test_includes_only_requested_families(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()
        # Seed BOTH families into the catalog...
        seed_feature_specs(repo, families=["alpha", "beta"], registry=registry)

        # ...but request only alpha — beta_feat must be excluded.
        specs = build_expected_specs(
            repo, feature_set="v1", families=["alpha"], registry=registry
        )

        keys = {s.feature_key for s in specs}
        assert keys == {_CRITICAL_KEY, "alpha_plain"}
        assert "beta_feat" not in keys

    def test_produces_feature_spec_value_objects(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()
        seed_feature_specs(repo, families=["beta"], registry=registry)

        specs = build_expected_specs(
            repo, feature_set="v1", families=["beta"], registry=registry
        )

        assert len(specs) == 1
        spec = specs[0]
        assert isinstance(spec, FeatureSpec)
        assert spec.feature_key == "beta_feat"
        assert spec.version == 1
        assert spec.dtype == "cat"
        assert spec.critical is False

    def test_empty_families_returns_empty(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()
        seed_feature_specs(repo, families=["alpha"], registry=registry)

        # No families requested -> no expected specs, even though the catalog
        # has seeded rows (lockstep: validate only against registered families).
        assert (
            build_expected_specs(
                repo, feature_set="v1", families=[], registry=registry
            )
            == ()
        )

    def test_unknown_family_raises(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        repo = _FakeFeatureSpecRepo()

        with pytest.raises(KeyError):
            build_expected_specs(
                repo, feature_set="v1", families=["nope"], registry=registry
            )

    def test_raises_on_catalog_drift_when_keys_not_seeded(
        self, registry: dict[str, tuple[FeatureSpecRow, ...]]
    ) -> None:
        # Registered family requested, but nothing was seeded -> the keys would
        # be silently dropped from expected_specs. Fail loud instead.
        repo = _FakeFeatureSpecRepo()

        with pytest.raises(FeatureContractError, match="catalog drift"):
            build_expected_specs(
                repo, feature_set="v1", families=["alpha"], registry=registry
            )


class TestProductionRegistry:
    def test_is_empty_in_r2(self) -> None:
        # Lockstep: no family is registered before its extractor exists (R3 = Elo).
        assert _REGISTRY == {}

    def test_critical_keyset_is_the_elo_base_ratings(self) -> None:
        # §0.5 / §15.6: minimal critical set — only never-NULL base Elo ratings.
        assert _CRITICAL_KEY in _CRITICAL_FEATURE_KEYS
        assert "p1_elo_pre" in _CRITICAL_FEATURE_KEYS
        # Derived reliability booleans are NOT critical (always present, but the
        # set is restricted to the rating keys carrying the 1500 fallback).
        assert "p1_elo_reliability_low" not in _CRITICAL_FEATURE_KEYS
