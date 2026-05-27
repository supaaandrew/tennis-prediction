"""Modeling Agent ML internals (M1).

One module per concern (mirrors `agents/research/features/`):
  - `feature_set`  — resolve the model feature set + `feature_hash` (§M20/§M21)
  - `assembly`     — `feature_matrix` + labels → `(X, y, meta)` (§M20a/§M21f)
  - `splits`       — walk-forward CV + tournament embargo + tail carve (§M20c/§M21c/e)
  - `base_learners`— XGB/LGBM + walk-forward per-learner OOF (§M20a/§M21d)
  - `noise`        — H1 forecast-noise injection (§M22, seeded)
  - `stacker`      — logistic meta-learner over the OOF matrix (M1b)
  - `calibration`  — Platt/isotonic tail calibration + degraded path (§M21c)
  - `edge`         — edge vs Shin/proportional implied probs (§M21a/C9)
  - `kelly`        — fractional Kelly + per-match/same-day caps (§M24/H12)
  - `metrics`      — ECE + Kelly-ROI backtest (M1b secondary metrics)
  - `artifacts`    — joblib persistence + version minting (§M20d/§M23)

Submodules are imported directly (some pull in pandas/xgboost), so this package
`__init__` is intentionally empty to keep import cost off unrelated paths.
"""
