"""Probability calibration on the held-out tail (M1b).

The calibrator maps the stacker's raw probability `p1_prob_raw` → `p1_prob_cal`,
fit on `split.tail_idx` ONLY (§M21c — the tail is carved before CV and never
enters a fold, so there is no OOF→calibration leak). `config.modeling.calibration`
supplies `method` (`platt`|`isotonic`), `tail_days`, and `min_calibration_samples`.

Degraded path (pre-step 5.1): when the tail has fewer than
`min_calibration_samples` rows (including an EMPTY tail, `len == 0`) OR is
single-class, a **passthrough (identity) calibrator** is installed and `degraded`
is set — the agent surfaces this as `partial` (code `calibration_degraded`, NOT a
hard failure; the model is still served, just uncalibrated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.models.base_learners import _positive_proba

_logger = get_logger("tennis.models.calibration")

_PASSTHROUGH = "passthrough"
_PLATT = "platt"
_ISOTONIC = "isotonic"
_MIN_CLASSES = 2
_MAX_ITER = 1000


@dataclass(frozen=True, slots=True)
class Calibrator:
    """A fitted probability calibrator (or a passthrough when degraded)."""

    method: str            # "platt" | "isotonic" | "passthrough"
    model: Any | None      # None ⟺ passthrough
    degraded: bool
    n_samples: int

    def predict(self, raw: np.ndarray) -> np.ndarray:
        """Map raw stacker probabilities → calibrated probabilities in [0, 1]."""
        raw = np.asarray(raw, dtype="float64")
        if self.model is None:  # passthrough: clip only
            return np.clip(raw, 0.0, 1.0)
        if self.method == _PLATT:
            return _positive_proba(self.model, raw.reshape(-1, 1))
        return np.clip(self.model.predict(raw), 0.0, 1.0)  # isotonic


def fit_calibrator(
    raw_preds: np.ndarray, y_tail: np.ndarray, *, config: AppConfig
) -> Calibrator:
    """Fit a calibrator on the tail's raw predictions vs labels.

    Returns a degraded passthrough calibrator when the tail is too small
    (`< min_calibration_samples`, including empty) or single-class — never raises.
    """
    raw_preds = np.asarray(raw_preds, dtype="float64")
    y_tail = np.asarray(y_tail)
    n = len(y_tail)
    min_samples = config.modeling.calibration.min_calibration_samples
    if n < min_samples or np.unique(y_tail).size < _MIN_CLASSES:
        _logger.warning(
            "calibration_degraded",
            n_samples=n,
            min_calibration_samples=min_samples,
            single_class=bool(np.unique(y_tail).size < _MIN_CLASSES),
        )
        return Calibrator(
            method=_PASSTHROUGH, model=None, degraded=True, n_samples=n
        )

    method = config.modeling.calibration.method
    if method == _ISOTONIC:
        model: Any = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw_preds, y_tail)
    else:  # platt (default)
        model = LogisticRegression(max_iter=_MAX_ITER)
        model.fit(raw_preds.reshape(-1, 1), y_tail)
    _logger.info("calibrator_fit", method=method, n_samples=n)
    return Calibrator(method=method, model=model, degraded=False, n_samples=n)
