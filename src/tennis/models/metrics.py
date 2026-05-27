"""Secondary modeling metrics (M1b): ECE + Kelly-ROI backtest.

These are diagnostic (`config.modeling.metrics.secondary`); `logloss` (primary)
and `brier` come from `base_learners._cv_metrics`. All are a TRAINING concept —
the live slate is unlabelled (C4), so prediction mode surfaces the STORED tail
metrics rather than recomputing (pre-step 6.2).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tennis.core.config import AppConfig

_DEFAULT_BINS = 10


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, *, n_bins: int = _DEFAULT_BINS
) -> float:
    """Equal-width-bin ECE: Σ_bin (n_bin/N)·|accuracy − confidence|.

    NaN for an empty input (no calibration evidence)."""
    y_true = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    n = len(y_true)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against interior edges → bin index in [0, n_bins-1].
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y_true[mask].mean())
        ece += abs(acc - conf) * count / n
    return float(ece)


def roi_kelly_backtest(
    *,
    probs: Sequence[float],
    y_true: Sequence[int],
    implied_p1: Sequence[float | None],
    config: AppConfig,
) -> float:
    """ROI of a fractional-Kelly strategy over labelled rows (a backtest signal).

    For each row the model probability `probs[i]` is staked against the row's
    `implied_p1[i]` (the chosen devig's p1 implied prob) using the SAME sizing as
    live (`per_match_kelly` math: per-match cap, no same-day cap — this spans many
    days), then settled at the fair decimal odds `1/implied`. ROI = Σpnl / Σstaked
    (0.0 when nothing was staked). Rows with no usable implied prob are skipped
    (C9). The two-outcome market lets exactly one side qualify per row.
    """
    frac = config.modeling.kelly.fraction
    cap = config.modeling.kelly.max_exposure_pct
    min_edge = config.modeling.edge.min_edge_to_bet
    pnl = 0.0
    staked = 0.0
    for p, y, imp in zip(probs, y_true, implied_p1, strict=True):
        if imp is None or not (0.0 < imp < 1.0):
            continue
        edge1 = p - imp
        edge2 = -edge1
        if edge1 >= min_edge:
            stake = min(frac * edge1 / (1.0 - imp), cap)
            staked += stake
            pnl += stake * ((1.0 / imp) - 1.0) if y == 1 else -stake
        elif edge2 >= min_edge:
            imp2 = 1.0 - imp  # p2 implied; denom 1-imp2 = imp
            stake = min(frac * edge2 / imp, cap)
            staked += stake
            pnl += stake * ((1.0 / imp2) - 1.0) if y == 0 else -stake
    return pnl / staked if staked > 0.0 else 0.0
