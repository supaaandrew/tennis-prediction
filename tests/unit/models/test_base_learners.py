"""Base learners + walk-forward CV (M1a) — §M20a native categorical, §M21d
early-stop on a train sub-split, OOF shape/metrics."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from tennis.core.config import AppConfig
from tennis.core.errors import InsufficientTrainingDataError
from tennis.models.assembly import assemble_training_data
from tennis.models.base_learners import (
    _cv_metrics,
    _positive_proba,
    chrono_eval_split,
    cross_val_oof,
    train_base_learners,
)
from tennis.models.feature_set import ModelFeatureSet, compute_feature_hash
from tennis.models.splits import WalkForwardSplit, build_walk_forward_splits
from tennis.storage.postgres.rows import FeatureMatrixRow, MatchRow

_KEYS_DTYPES = {
    "elo_diff_blended": "float",
    "p1_elo_pre": "float",
    "surface_transition_type": "cat",
}


def _fs() -> ModelFeatureSet:
    keys = tuple(sorted(_KEYS_DTYPES))
    return ModelFeatureSet(
        keys=keys,
        categorical_keys=frozenset({"surface_transition_type"}),
        dtype_by_key=dict(_KEYS_DTYPES),
        feature_hash=compute_feature_hash(keys),
    )


def _shrink(config: AppConfig) -> AppConfig:
    md = config.modeling
    return config.model_copy(
        update={
            "modeling": md.model_copy(
                update={
                    "splits": md.splits.model_copy(
                        update={"min_train_seasons": 1, "n_folds": 2}
                    ),
                    "calibration": md.calibration.model_copy(update={"tail_days": 1}),
                    "base_learners": md.base_learners.model_copy(
                        update={
                            "xgb": md.base_learners.xgb.model_copy(
                                update={
                                    "n_estimators": 8,
                                    "early_stopping_rounds": 3,
                                    "max_depth": 2,
                                }
                            ),
                            "lgbm": md.base_learners.lgbm.model_copy(
                                update={
                                    "n_estimators": 8,
                                    "early_stopping_rounds": 3,
                                    "num_leaves": 5,
                                }
                            ),
                        }
                    ),
                }
            )
        }
    )


def _dataset(seasons: int = 4, per_season: int = 16, start: int = 2016):
    matches: list[MatchRow] = []
    frows: list[FeatureMatrixRow] = []
    i = 0
    trans = ["same", "clay->hard", "none"]
    for s in range(seasons):
        y = start + s
        for k in range(per_season):
            mid = 1000 + i
            p1win = k % 2 == 0  # balanced within every season
            matches.append(
                MatchRow(
                    match_id=mid, tournament_id=y * 1000, round="R16",
                    match_date=date(y, 6, 1 + k), p1_id=1 + i, p2_id=9000 + i,
                    status="final", source="t", source_uid=f"u{i}",
                    winner_id=(1 + i) if p1win else (9000 + i),
                )
            )
            frows.append(
                FeatureMatrixRow(
                    match_id=mid, feature_set="v1",
                    as_of_ts=datetime(y, 5, 1, tzinfo=UTC),
                    payload={
                        "elo_diff_blended": (2.0 if p1win else -2.0) + (k % 3) * 0.1,
                        "p1_elo_pre": 1500.0 + i,
                        "surface_transition_type": trans[k % 3],
                    },
                )
            )
            i += 1
    return assemble_training_data(matches=matches, feature_rows=frows, feature_set=_fs())


class TestChronoEvalSplit:
    def test_eval_is_most_recent_slice(self):
        dates = [date(2020, 1, d) for d in [5, 1, 3, 2, 4, 6, 7, 8, 9, 10]]
        res = chrono_eval_split(dates)
        assert res is not None
        fit, ev = res
        assert sorted([*fit, *ev]) == list(range(10))
        assert set(fit).isdisjoint(ev)
        latest = sorted(range(10), key=lambda i: dates[i])[-len(ev) :]
        assert set(ev) == set(latest)

    def test_none_when_too_few_rows(self):
        assert chrono_eval_split([date(2020, 1, 1)] * 5) is None

    def test_keeps_at_least_one_fit_row(self):
        res = chrono_eval_split([date(2020, 1, d + 1) for d in range(8)])
        assert res is not None
        fit, ev = res
        assert len(fit) >= 1 and len(ev) >= 1


class TestTraining:
    def test_trains_with_categorical_and_nan(self, real_config):
        cfg = _shrink(real_config)
        ds = _dataset()
        rem = list(range(ds.n_rows))
        trained = train_base_learners(
            ds.X.iloc[rem], ds.y.iloc[rem], [ds.dates[i] for i in rem],
            categorical_keys=_fs().categorical_keys, config=cfg,
        )
        assert trained.xgb is not None and trained.lgbm is not None

    def test_xgb_enable_categorical(self, real_config):
        cfg = _shrink(real_config)
        ds = _dataset()
        trained = train_base_learners(
            ds.X, ds.y, list(ds.dates),
            categorical_keys=_fs().categorical_keys, config=cfg,
        )
        assert trained.xgb.get_params()["enable_categorical"] is True

    def test_predict_p1_returns_probabilities(self, real_config):
        cfg = _shrink(real_config)
        ds = _dataset()
        trained = train_base_learners(
            ds.X, ds.y, list(ds.dates),
            categorical_keys=_fs().categorical_keys, config=cfg,
        )
        out = trained.predict_p1(ds.X.iloc[:5])
        assert set(out) == {"xgb", "lgbm", "mean"}
        for arr in out.values():
            assert len(arr) == 5
            assert np.all((arr >= 0.0) & (arr <= 1.0))


class TestCrossVal:
    def test_oof_covers_only_test_rows(self, real_config):
        cfg = _shrink(real_config)
        ds = _dataset()
        split = build_walk_forward_splits(
            dates=ds.dates, tournament_ids=ds.tournament_ids,
            n_folds=2, min_train_seasons=1, tail_days=1,
        )
        cv = cross_val_oof(
            ds.X, ds.y, ds.dates, split,
            categorical_keys=_fs().categorical_keys, config=cfg,
        )
        all_test = {i for _, te in split.folds for i in te}
        assert set(cv.oof_index) <= all_test
        assert len(cv.oof_mean) == len(cv.oof_index)

    def test_per_learner_oof_aligned(self, real_config):
        # M1b: per-learner OOF columns feed the stacker; both align with oof_index
        # and the (n,2) design matrix has the right shape.
        cfg = _shrink(real_config)
        ds = _dataset()
        split = build_walk_forward_splits(
            dates=ds.dates, tournament_ids=ds.tournament_ids,
            n_folds=2, min_train_seasons=1, tail_days=1,
        )
        cv = cross_val_oof(
            ds.X, ds.y, ds.dates, split,
            categorical_keys=_fs().categorical_keys, config=cfg,
        )
        n = len(cv.oof_index)
        assert len(cv.oof_xgb) == n and len(cv.oof_lgbm) == n
        assert cv.oof_matrix().shape == (n, 2)
        assert np.all((cv.oof_xgb >= 0.0) & (cv.oof_xgb <= 1.0))

    def test_oof_metrics_finite(self, real_config):
        cfg = _shrink(real_config)
        ds = _dataset()
        split = build_walk_forward_splits(
            dates=ds.dates, tournament_ids=ds.tournament_ids,
            n_folds=2, min_train_seasons=1, tail_days=1,
        )
        cv = cross_val_oof(
            ds.X, ds.y, ds.dates, split,
            categorical_keys=_fs().categorical_keys, config=cfg,
        )
        assert cv.metrics["n_oof"] > 0
        assert np.isfinite(cv.metrics["logloss"])
        assert np.isfinite(cv.metrics["brier"])

    def test_empty_oof_metrics_are_nan(self):
        m = _cv_metrics(np.asarray([], dtype="int64"), np.asarray([], dtype="float64"))
        assert m["n_oof"] == 0
        assert np.isnan(m["logloss"])


class TestDegenerateFolds:
    """Codex HIGH: single-class folds must not crash; they skip + count."""

    def _tiny(self):
        # rows 0-2 are class 1 only; rows 3-5 mix. Categorical column present.
        X = pd.DataFrame(
            {
                "f": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "surface_transition_type": pd.Series(["same"] * 6, dtype="category"),
            }
        )
        y = pd.Series([1, 1, 1, 0, 1, 0], dtype="int64")
        dates = [date(2018, 1, 1)] * 6
        # fold0 train [0,1,2] is single-class -> skip; fold1 train [0,1,2,3] mixes.
        split = WalkForwardSplit(
            folds=(((0, 1, 2), (3,)), ((0, 1, 2, 3), (4, 5))),
            tail_idx=(),
            remainder_idx=(0, 1, 2, 3, 4, 5),
        )
        return X, y, dates, split

    def test_single_class_fold_skipped_and_counted(self, real_config):
        cfg = _shrink(real_config)
        X, y, dates, split = self._tiny()
        cv = cross_val_oof(
            X, y, dates, split,
            categorical_keys=frozenset({"surface_transition_type"}), config=cfg,
        )
        assert cv.degenerate_folds == 1
        assert 3 not in cv.oof_index          # fold0 (single-class) skipped
        assert set(cv.oof_index) == {4, 5}    # only fold1 contributed OOF

    def test_heartbeat_called_per_fold(self, real_config):
        cfg = _shrink(real_config)
        X, y, dates, split = self._tiny()
        beats: list[int] = []
        cross_val_oof(
            X, y, dates, split,
            categorical_keys=frozenset({"surface_transition_type"}), config=cfg,
            heartbeat=lambda: beats.append(1),
        )
        assert len(beats) == len(split.folds)  # one beat per fold (§L7)

    def test_single_class_final_train_raises(self, real_config):
        cfg = _shrink(real_config)
        ds = _dataset()
        y_one = pd.Series([1] * ds.n_rows, dtype="int64")
        with pytest.raises(InsufficientTrainingDataError):
            train_base_learners(
                ds.X, y_one, list(ds.dates),
                categorical_keys=_fs().categorical_keys, config=cfg,
            )


class TestPositiveProba:
    """Codex HIGH: a one-class model's (n,1) predict_proba must not IndexError."""

    class _OneClass:
        def __init__(self, label):
            self.classes_ = np.array([label])

        def predict_proba(self, x):
            return np.ones((len(x), 1))

    def test_one_column_class1_returns_ones(self):
        out = _positive_proba(self._OneClass(1), pd.DataFrame({"a": [1, 2, 3]}))
        assert list(out) == [1.0, 1.0, 1.0]

    def test_one_column_class0_returns_zeros(self):
        out = _positive_proba(self._OneClass(0), pd.DataFrame({"a": [1, 2]}))
        assert list(out) == [0.0, 0.0]
