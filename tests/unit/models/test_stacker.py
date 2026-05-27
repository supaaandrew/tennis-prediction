"""Stacking meta-learner (M1b)."""

from __future__ import annotations

import numpy as np
import pytest

from tennis.core.errors import InsufficientTrainingDataError
from tennis.models.stacker import TrainedStacker, train_stacker


def _oof(n: int = 200, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 1.0, size=(n, 2))
    y = (matrix[:, 0] + matrix[:, 1] > 1.0).astype(int)
    return matrix, y


class TestTrain:
    def test_fits_and_predicts_in_unit_interval(self, real_config):
        matrix, y = _oof()
        stacker = train_stacker(oof_matrix=matrix, y_oof=y, config=real_config)
        assert isinstance(stacker, TrainedStacker)
        p = stacker.predict_p1({"xgb": matrix[:, 0], "lgbm": matrix[:, 1]})
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert len(p) == len(y)

    def test_single_class_oof_raises(self, real_config):
        # §M22/pre-step 5.2: logistic cannot fit one class → loud failure.
        matrix, _ = _oof()
        with pytest.raises(InsufficientTrainingDataError, match="single-class"):
            train_stacker(
                oof_matrix=matrix, y_oof=np.zeros(len(matrix), dtype=int),
                config=real_config,
            )
