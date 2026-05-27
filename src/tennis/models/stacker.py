"""Stacking meta-learner (M1b).

A `LogisticRegression` over the per-learner out-of-fold matrix `[oof_xgb,
oof_lgbm]` (§M21c — the OOF rows never include the calibration tail, so the
stacker is fit only on the older remainder via the walk-forward folds). The
combined output is the raw probability `p1_prob_raw`; the calibrator maps it to
`p1_prob_cal` downstream.

`config.modeling.meta_learner` supplies the hyperparameters (`type="logistic"`,
`C=1.0`). A single-class OOF label set raises `InsufficientTrainingDataError`
(logistic regression cannot fit one class) — mirroring the base-learner guard;
this is the stacker's only size precondition (the agent's zero-OOF/NaN gate is
the empty-OOF floor — §M22/pre-step note: NOT coupled to
`min_calibration_samples`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from tennis.core.config import AppConfig
from tennis.core.errors import InsufficientTrainingDataError
from tennis.core.logging import get_logger
from tennis.models.base_learners import _positive_proba

_logger = get_logger("tennis.models.stacker")

_MIN_CLASSES = 2
_MAX_ITER = 1000


@dataclass(frozen=True, slots=True)
class TrainedStacker:
    """A fitted logistic meta-learner over the `[xgb, lgbm]` probability pair."""

    model: Any

    def predict_p1(self, base_preds: Mapping[str, np.ndarray]) -> np.ndarray:
        """`p1_prob_raw` from a base-learner prediction dict (`predict_p1` output:
        keys `xgb`/`lgbm`)."""
        matrix = np.column_stack([base_preds["xgb"], base_preds["lgbm"]])
        return _positive_proba(self.model, matrix)


def train_stacker(
    *, oof_matrix: np.ndarray, y_oof: np.ndarray, config: AppConfig
) -> TrainedStacker:
    """Fit the logistic stacker on the per-learner OOF design matrix.

    Raises `InsufficientTrainingDataError` if `y_oof` has fewer than two outcome
    classes — `LogisticRegression` cannot fit a single class (mirrors the
    base-learner single-class guard).
    """
    if np.unique(y_oof).size < _MIN_CLASSES:
        raise InsufficientTrainingDataError(
            "stacker OOF labels are single-class; cannot fit the meta-learner"
        )
    model = LogisticRegression(C=config.modeling.meta_learner.C, max_iter=_MAX_ITER)
    model.fit(oof_matrix, y_oof)
    _logger.info("stacker_fit", n_oof=len(y_oof), C=config.modeling.meta_learner.C)
    return TrainedStacker(model=model)
