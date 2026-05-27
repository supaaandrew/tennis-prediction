"""Base learners + walk-forward CV (M1a).

XGBoost + LightGBM, both consuming the assembled frame directly:

  - **§M20a native categorical:** XGBoost with `enable_categorical=True`,
    LightGBM with `categorical_feature=` — pandas `category` columns are used as
    categoricals with no one-hot expansion.
  - **Native missing:** `NaN` is passed straight through; the §M8 sparse-NULL
    signal is never imputed away.
  - **§M21d early stopping on a TRAIN sub-split:** each fit carves the most
    recent `_EARLY_STOP_EVAL_FRACTION` of its OWN train block (chronologically)
    as the early-stopping eval set — never the held-out test/OOF fold — so the
    stopping signal cannot leak into the OOF predictions that train the M1b
    stacker.

M1a uses the simple mean of the two base learners as the model probability for
CV metrics; M1b replaces it with the logistic stacker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from tennis.core.config import AppConfig
from tennis.core.errors import InsufficientTrainingDataError
from tennis.core.logging import get_logger
from tennis.models.splits import WalkForwardSplit

_logger = get_logger("tennis.models.base_learners")

# §M21d: fraction of each train block (most-recent by date) used as the
# early-stopping eval set. Structural constant (not config) — it is a leakage
# guard, not a tunable.
_EARLY_STOP_EVAL_FRACTION = 0.15
# Below this many train rows, skip early stopping (eval split would be degenerate).
_MIN_ROWS_FOR_EARLY_STOP = 8
# A classifier needs both outcome classes present to fit; a single-class train
# block crashes XGBoost (ValueError) and yields a 1-column predict_proba.
_MIN_CLASSES = 2


@dataclass(frozen=True, slots=True)
class TrainedBaseLearners:
    """A fitted XGB + LGBM pair over one train block."""

    xgb: Any
    lgbm: Any

    def predict_p1(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """P(p1 wins) from each base learner, plus their mean."""
        xgb_p = _positive_proba(self.xgb, X)
        lgbm_p = _positive_proba(self.lgbm, X)
        return {"xgb": xgb_p, "lgbm": lgbm_p, "mean": (xgb_p + lgbm_p) / 2.0}


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """P(class==1) robust to a one-class model whose `predict_proba` is `(n,1)`.

    Single-class fits are guarded against in `train_base_learners`, so this is
    defensive: a one-class model returns a constant 1.0/0.0 for the only class
    rather than an IndexError on `[:, 1]`.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.full(len(X), 1.0 if model.classes_[0] == 1 else 0.0)
    return proba[:, 1]


@dataclass(frozen=True, slots=True)
class CVResult:
    """Out-of-fold predictions (for the M1b stacker) + CV metrics.

    `oof_xgb`/`oof_lgbm` are the per-learner OOF columns — the M1b stacker's
    training features (`LogisticRegression` over `[oof_xgb, oof_lgbm]`). `oof_mean`
    is their average, the M1a CV-metrics probability. All three are row-aligned
    with `oof_index`.
    """

    oof_index: tuple[int, ...]
    oof_mean: np.ndarray
    oof_xgb: np.ndarray
    oof_lgbm: np.ndarray
    metrics: Mapping[str, float]
    n_folds: int
    degenerate_folds: int = 0

    def oof_matrix(self) -> np.ndarray:
        """The `(n_oof, 2)` per-learner OOF design matrix for the stacker."""
        return np.column_stack([self.oof_xgb, self.oof_lgbm])


def chrono_eval_split(dates_train: Sequence[date]) -> tuple[list[int], list[int]] | None:
    """Carve the most-recent `_EARLY_STOP_EVAL_FRACTION` of a train block as the
    early-stop eval set (§M21d). Returns `(fit_pos, eval_pos)` as positions into
    the train block, or `None` when there are too few rows for early stopping.
    """
    n = len(dates_train)
    if n < _MIN_ROWS_FOR_EARLY_STOP:
        return None
    order = sorted(range(n), key=lambda i: dates_train[i])
    n_eval = max(1, round(_EARLY_STOP_EVAL_FRACTION * n))
    n_eval = min(n_eval, n - 1)  # keep ≥1 fit row
    return order[: n - n_eval], order[n - n_eval :]


def _xgb_params(config: AppConfig) -> dict[str, Any]:
    c = config.modeling.base_learners.xgb
    return {
        "n_estimators": c.n_estimators,
        "max_depth": c.max_depth,
        "learning_rate": c.learning_rate,
        "subsample": c.subsample,
        "colsample_bytree": c.colsample_bytree,
        "reg_lambda": c.reg_lambda,
        "tree_method": c.tree_method,
        "n_jobs": c.n_jobs,
        "enable_categorical": True,  # §M20a
    }


def _lgbm_params(config: AppConfig) -> dict[str, Any]:
    c = config.modeling.base_learners.lgbm
    return {
        "n_estimators": c.n_estimators,
        "num_leaves": c.num_leaves,
        "learning_rate": c.learning_rate,
        "feature_fraction": c.feature_fraction,
        "bagging_fraction": c.bagging_fraction,
        "bagging_freq": 1,  # required for bagging_fraction to take effect
        "lambda_l2": c.lambda_l2,
        "n_jobs": c.n_jobs,
        "verbosity": -1,
    }


def train_base_learners(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: Sequence[date],
    *,
    categorical_keys: frozenset[str],
    config: AppConfig,
) -> TrainedBaseLearners:
    """Fit XGB + LGBM on one train block with a §M21d early-stop sub-split.

    Raises `InsufficientTrainingDataError` if the train block is single-class —
    a classifier cannot fit (XGBoost raises, and a one-class fit breaks
    `predict_proba[:, 1]`). CV callers pre-skip degenerate folds, so this guard
    bites only the FINAL-model train path (→ agent fails cleanly, zero writes).
    """
    if y_train.nunique() < _MIN_CLASSES:
        raise InsufficientTrainingDataError(
            "training block has a single outcome class; cannot fit a classifier"
        )
    split = chrono_eval_split(dates_train)
    es_rounds_xgb = config.modeling.base_learners.xgb.early_stopping_rounds
    es_rounds_lgbm = config.modeling.base_learners.lgbm.early_stopping_rounds
    cat = sorted(categorical_keys)

    if split is None:
        xgb = XGBClassifier(**_xgb_params(config))
        xgb.fit(X_train, y_train, verbose=False)
        lgbm = LGBMClassifier(**_lgbm_params(config))
        lgbm.fit(X_train, y_train, categorical_feature=cat or "auto")
        return TrainedBaseLearners(xgb=xgb, lgbm=lgbm)

    fit_pos, eval_pos = split
    X_fit, y_fit = X_train.iloc[fit_pos], y_train.iloc[fit_pos]
    X_eval, y_eval = X_train.iloc[eval_pos], y_train.iloc[eval_pos]

    xgb = XGBClassifier(early_stopping_rounds=es_rounds_xgb, **_xgb_params(config))
    xgb.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=False)

    lgbm = LGBMClassifier(**_lgbm_params(config))
    lgbm.fit(
        X_fit,
        y_fit,
        eval_set=[(X_eval, y_eval)],
        categorical_feature=cat or "auto",
        callbacks=[early_stopping(es_rounds_lgbm, verbose=False), log_evaluation(0)],
    )
    return TrainedBaseLearners(xgb=xgb, lgbm=lgbm)


def cross_val_oof(
    X: pd.DataFrame,
    y: pd.Series,
    dates: Sequence[date],
    split: WalkForwardSplit,
    *,
    categorical_keys: frozenset[str],
    config: AppConfig,
    heartbeat: Callable[[], None] | None = None,
) -> CVResult:
    """Walk-forward CV: per fold, fit on the fold's train block (with the §M21d
    eval sub-split) and predict its test block → OOF mean predictions + metrics.

    `heartbeat` (the agent threads `ctx.heartbeat`) fires once per fold so a
    multi-fold, 800-estimator train cannot exceed `orphan_after_s` with no beat
    (§L7). A fold whose train/test block is empty after embargo, or whose train
    block is single-class, is SKIPPED and counted (`degenerate_folds`) — never
    fed to a classifier that would crash.
    """
    oof_pos: list[int] = []
    oof_pred: list[float] = []
    oof_xgb: list[float] = []
    oof_lgbm: list[float] = []
    degenerate = 0
    for train_idx, test_idx in split.folds:
        if heartbeat is not None:
            heartbeat()
        if not train_idx or not test_idx:
            degenerate += 1
            continue
        if y.iloc[list(train_idx)].nunique() < _MIN_CLASSES:
            # single-class fold (sparse/embargoed) — cannot fit; skip + count.
            degenerate += 1
            continue
        trained = train_base_learners(
            X.iloc[list(train_idx)],
            y.iloc[list(train_idx)],
            [dates[i] for i in train_idx],
            categorical_keys=categorical_keys,
            config=config,
        )
        preds = trained.predict_p1(X.iloc[list(test_idx)])
        oof_pos.extend(test_idx)
        oof_pred.extend(preds["mean"].tolist())
        oof_xgb.extend(preds["xgb"].tolist())
        oof_lgbm.extend(preds["lgbm"].tolist())

    oof_mean = np.asarray(oof_pred, dtype="float64")
    y_oof = y.iloc[oof_pos].to_numpy() if oof_pos else np.asarray([], dtype="int64")
    metrics = _cv_metrics(y_oof, oof_mean)
    _logger.info(
        "cross_val_oof_complete",
        folds=len(split.folds), degenerate_folds=degenerate, oof=len(oof_pos), **metrics,
    )
    return CVResult(
        oof_index=tuple(oof_pos),
        oof_mean=oof_mean,
        oof_xgb=np.asarray(oof_xgb, dtype="float64"),
        oof_lgbm=np.asarray(oof_lgbm, dtype="float64"),
        metrics=metrics,
        n_folds=len(split.folds),
        degenerate_folds=degenerate,
    )


def _cv_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """logloss (primary) + brier (secondary) over the OOF rows."""
    if len(y_true) == 0:
        return {"logloss": float("nan"), "brier": float("nan"), "n_oof": 0}
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "logloss": float(log_loss(y_true, p_clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_clipped)),
        "n_oof": len(y_true),
    }
