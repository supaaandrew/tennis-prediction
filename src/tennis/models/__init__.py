"""Modeling Agent ML internals (M1).

One module per concern (mirrors `agents/research/features/`):
  - `feature_set`  — resolve the model feature set + `feature_hash` (§M20/§M21)
  - `assembly`     — `feature_matrix` + labels → `(X, y, meta)` (§M20a/§M21f)
  - `splits`       — walk-forward CV + tournament embargo + tail carve (§M20c/§M21c/e)
  - `base_learners`— XGB/LGBM + walk-forward OOF (§M20a/§M21d)
  - `noise`        — H1 forecast-noise injection (M1a stub)
  - `artifacts`    — joblib persistence + version minting (§M20d)

Submodules are imported directly (some pull in pandas/xgboost), so this package
`__init__` is intentionally empty to keep import cost off unrelated paths.
"""
