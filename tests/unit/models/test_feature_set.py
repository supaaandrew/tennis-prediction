"""Model feature-set resolution (M1a) — §M20a/b/d, §M21a/b exclusions + hash."""

from __future__ import annotations

import pytest

from tennis.agents.research import specs
from tennis.agents.research.specs import family_feature_keys
from tennis.models.feature_set import (
    _BACKTEST_ONLY_KEYS,
    ModelFeatureSet,
    compute_feature_hash,
    excluded_model_keys,
    resolve_model_feature_set,
)
from tennis.storage.postgres.rows import FeatureSpecRow


class TestExclusions:
    def test_market_family_never_in_model_x(self, active_specs):
        """§M21a regression: NO key from the `market` family reaches model X."""
        fs = resolve_model_feature_set(active_specs)
        market = family_feature_keys("market")
        assert market, "sanity: market family is non-empty"
        assert market.isdisjoint(set(fs.keys))

    def test_future_market_key_auto_excluded(self, active_specs, monkeypatch):
        """The exclusion is family-derived, so a NEW market key added by a future
        extractor is dropped without touching the modeling code (clarification #3)."""
        new_key = FeatureSpecRow("p1_implied_newbook_decision", 1, "float")
        monkeypatch.setitem(
            specs._REGISTRY, "market", specs._REGISTRY["market"] + (new_key,)
        )
        fs = resolve_model_feature_set([*active_specs, new_key])
        assert "p1_implied_newbook_decision" not in fs.keys

    def test_forecast_uncertainty_bucket_excluded(self, active_specs):
        """§M21b: the noise-control bucket is never a model feature."""
        fs = resolve_model_feature_set(active_specs)
        assert "forecast_uncertainty_bucket" not in fs.keys

    def test_backtest_only_keys_excluded(self, active_specs):
        """§M20b: closing-derived keys never reach model X."""
        fs = resolve_model_feature_set(active_specs)
        assert _BACKTEST_ONLY_KEYS.isdisjoint(set(fs.keys))

    def test_excluded_set_covers_market_bucket_and_backtest(self):
        excluded = excluded_model_keys()
        assert family_feature_keys("market") <= excluded
        assert "forecast_uncertainty_bucket" in excluded
        assert excluded >= _BACKTEST_ONLY_KEYS

    def test_non_market_keys_survive(self, active_specs):
        """Sanity: legitimate predictors (elo, surface) are kept."""
        fs = resolve_model_feature_set(active_specs)
        assert "elo_diff_blended" in fs.keys
        assert "surface_transition_type" in fs.keys


class TestDtypesAndKeys:
    def test_keys_sorted_deterministic(self, active_specs):
        fs = resolve_model_feature_set(active_specs)
        assert list(fs.keys) == sorted(fs.keys)

    def test_categorical_keys_are_cat_dtype_only(self, active_specs):
        fs = resolve_model_feature_set(active_specs)
        # surface_transition_type is the only surviving `cat` key (bucket excluded).
        assert fs.categorical_keys == frozenset({"surface_transition_type"})
        for k in fs.categorical_keys:
            assert fs.dtype_by_key[k] == "cat"

    def test_dtype_map_covers_all_keys(self, active_specs):
        fs = resolve_model_feature_set(active_specs)
        assert set(fs.dtype_by_key) == set(fs.keys)


class TestFeatureHash:
    def test_hash_is_order_independent(self):
        a = compute_feature_hash(["b", "a", "c"])
        b = compute_feature_hash(["c", "b", "a"])
        assert a == b

    def test_hash_changes_when_keys_change(self):
        base = compute_feature_hash(["a", "b"])
        more = compute_feature_hash(["a", "b", "c"])
        assert base != more

    def test_resolved_hash_matches_helper(self, active_specs):
        fs = resolve_model_feature_set(active_specs)
        assert fs.feature_hash == compute_feature_hash(fs.keys)

    def test_hash_is_sha256_hex(self, active_specs):
        fs = resolve_model_feature_set(active_specs)
        assert len(fs.feature_hash) == 64
        int(fs.feature_hash, 16)  # valid hex


class TestGuards:
    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="zero feature keys"):
            ModelFeatureSet(
                keys=(), categorical_keys=frozenset(), dtype_by_key={}, feature_hash="x"
            )

    def test_all_excluded_raises(self):
        """A catalog of only market keys resolves to an empty model set → loud."""
        market_only = [
            FeatureSpecRow(k, 1, "float") for k in family_feature_keys("market")
        ]
        with pytest.raises(ValueError, match="zero feature keys"):
            resolve_model_feature_set(market_only)
