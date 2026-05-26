"""Artifact persistence + version minting (M1a) — §M20d."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import pytest

from tennis.models.artifacts import (
    TrainedModel,
    artifact_path,
    hyperparams_snapshot,
    load_artifact,
    mint_version,
    save_artifact,
)
from tennis.models.base_learners import TrainedBaseLearners
from tennis.models.feature_set import ModelFeatureSet, compute_feature_hash

_NOW = datetime(2026, 5, 26, 6, 30, 0, tzinfo=UTC)


def _model() -> TrainedModel:
    keys = ("a", "b")
    fs = ModelFeatureSet(
        keys=keys, categorical_keys=frozenset(),
        dtype_by_key={"a": "float", "b": "float"},
        feature_hash=compute_feature_hash(keys),
    )
    # Dummy picklable base learners — artifact IO is independent of model type.
    bl = TrainedBaseLearners(xgb={"x": 1}, lgbm={"l": 2})
    return TrainedModel(base_learners=bl, feature_set=fs, algo="xgb+lgbm_base")


class TestVersion:
    def test_format(self):
        # microsecond resolution (Codex M4): %f appends 6 digits (000000 here).
        v = mint_version(
            data_window_start=date(2017, 3, 1),
            data_window_end=date(2020, 11, 9),
            now=_NOW,
        )
        assert v == "2017-2020-20260526T063000000000Z"

    def test_distinct_within_same_second(self):
        # Two retrains in the same wall-clock SECOND must not collide on the
        # version PK (Codex M4) — microsecond resolution disambiguates them.
        base = datetime(2026, 5, 26, 6, 30, 0, tzinfo=UTC)
        v1 = mint_version(
            data_window_start=date(2017, 1, 1), data_window_end=date(2020, 1, 1),
            now=base,
        )
        v2 = mint_version(
            data_window_start=date(2017, 1, 1), data_window_end=date(2020, 1, 1),
            now=base.replace(microsecond=1),
        )
        assert v1 != v2

    def test_unique_per_instant(self):
        v1 = mint_version(
            data_window_start=date(2017, 1, 1), data_window_end=date(2020, 1, 1),
            now=_NOW,
        )
        v2 = mint_version(
            data_window_start=date(2017, 1, 1), data_window_end=date(2020, 1, 1),
            now=datetime(2026, 5, 26, 6, 31, 0, tzinfo=UTC),
        )
        assert v1 != v2


class TestPersistence:
    def test_artifact_path_under_dir(self):
        p = artifact_path("some/dir", "2017-2020-X")
        assert p == Path("some/dir") / "2017-2020-X.joblib"

    def test_save_load_roundtrip(self, tmp_path):
        model = _model()
        uri = save_artifact(model, artifact_dir=str(tmp_path), version="v1")
        loaded = load_artifact(uri)
        assert loaded.feature_set.feature_hash == model.feature_set.feature_hash
        assert loaded.feature_set.keys == model.feature_set.keys
        assert loaded.algo == "xgb+lgbm_base"

    def test_save_creates_nested_dir(self, tmp_path):
        target = tmp_path / "nested" / "models"
        uri = save_artifact(_model(), artifact_dir=str(target), version="v1")
        assert Path(uri).is_file()

    def test_load_wrong_type_raises(self, tmp_path):
        bad = tmp_path / "bad.joblib"
        joblib.dump({"not": "a model"}, bad)
        with pytest.raises(TypeError, match="not a TrainedModel"):
            load_artifact(str(bad))


class TestHyperparams:
    def test_snapshot_shape(self, real_config):
        snap = hyperparams_snapshot(real_config)
        assert set(snap) == {"xgb", "lgbm"}
        assert snap["xgb"]["n_estimators"] == real_config.modeling.base_learners.xgb.n_estimators
        assert "num_leaves" in snap["lgbm"]
