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
    def test_elo_family_registered_in_r3(self) -> None:
        # Lockstep: a family is registered only once its extractor exists. R3
        # landed "elo" (the first family) with its 9 §15.5 keys.
        assert "elo" in _REGISTRY
        elo_keys = {row.feature_key for row in _REGISTRY["elo"]}
        assert elo_keys == {
            "p1_elo_pre",
            "p2_elo_pre",
            "p1_elo_surface_pre",
            "p2_elo_surface_pre",
            "p1_elo_blended_pre",
            "p2_elo_blended_pre",
            "elo_diff_blended",
            "p1_elo_reliability_low",
            "p2_elo_reliability_low",
        }

    def test_critical_keyset_is_the_elo_base_ratings(self) -> None:
        # §0.5 / §15.6: minimal critical set — only never-NULL base Elo ratings.
        assert _CRITICAL_KEY in _CRITICAL_FEATURE_KEYS
        assert "p1_elo_pre" in _CRITICAL_FEATURE_KEYS
        # Derived reliability booleans are NOT critical (always present, but the
        # set is restricted to the rating keys carrying the 1500 fallback).
        assert "p1_elo_reliability_low" not in _CRITICAL_FEATURE_KEYS

    def test_r4_families_registered(self) -> None:
        # R4 appended rankings/form/h2h; R5a adds surface/serve_return — all in
        # lockstep with their extractors.
        assert set(_REGISTRY) == {
            "elo",
            "rankings",
            "form",
            "h2h",
            "surface",
            "serve_return",
        }
        assert {row.feature_key for row in _REGISTRY["rankings"]} == {
            "p1_rank_pre",
            "p2_rank_pre",
            "rank_diff",
            "p1_rank_stale",
            "p2_rank_stale",
        }
        assert {row.feature_key for row in _REGISTRY["h2h"]} == {
            "h2h_matches",
            "h2h_p1_wins",
            "h2h_p1_win_rate",
            "h2h_surface_matches",
            "h2h_surface_p1_win_rate",
            "h2h_win_rate_confidence",
            "h2h_win_rate_weighted",
        }
        # Form = 5 windows × (2 win_rate + 2 matches_played + 1 diff) = 25 rows.
        assert len(_REGISTRY["form"]) == 25

    def test_r5a_families_registered(self) -> None:
        # R5a: surface = the locked 7-key §15.5 catalog (transition keys are
        # single, p1-perspective per §M13); serve_return = its 15 §15.5 rows.
        assert {row.feature_key for row in _REGISTRY["surface"]} == {
            "p1_career_win_rate_surface",
            "p2_career_win_rate_surface",
            "p1_recent_win_rate_surface_365d",
            "p2_recent_win_rate_surface_365d",
            "surface_affinity_diff",
            "surface_transition_type",
            "surface_transition_exposure",
        }
        assert len(_REGISTRY["surface"]) == 7
        assert {row.feature_key for row in _REGISTRY["serve_return"]} == {
            "p1_first_serve_pct_career",
            "p2_first_serve_pct_career",
            "p1_first_serve_pct_365d",
            "p2_first_serve_pct_365d",
            "p1_first_serve_win_pct_365d",
            "p2_first_serve_win_pct_365d",
            "p1_second_serve_win_pct_365d",
            "p2_second_serve_win_pct_365d",
            "p1_ace_rate_365d",
            "p2_ace_rate_365d",
            "p1_df_rate_365d",
            "p2_df_rate_365d",
            "p1_bp_save_pct_365d",
            "p2_bp_save_pct_365d",
            "serve_dominance_diff_365d",
        }
        assert len(_REGISTRY["serve_return"]) == 15
        # The single transition keys carry the documented dtypes (cat/float).
        by_key = {row.feature_key: row for row in _REGISTRY["surface"]}
        assert by_key["surface_transition_type"].dtype == "cat"
        assert by_key["surface_transition_exposure"].dtype == "float"

    def test_r4_r5a_families_round_trip_and_non_critical(self) -> None:
        # Seed + build against the PRODUCTION registry: every registered key must
        # round-trip, and none of the R4/R5a families may be critical (§0.5/§M8).
        repo = _FakeFeatureSpecRepo()
        families = ["rankings", "form", "h2h", "surface", "serve_return"]
        seed_feature_specs(repo, families=families)
        specs = build_expected_specs(repo, feature_set="v1", families=families)

        expected = {
            row.feature_key for fam in families for row in _REGISTRY[fam]
        }
        assert {s.feature_key for s in specs} == expected
        assert all(s.critical is False for s in specs)
