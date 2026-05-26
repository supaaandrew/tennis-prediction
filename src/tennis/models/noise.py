"""H1 forecast-noise injection — M1a STUB.

H1: forecast-noise injection lives in the Modeling training loop ONLY, applied
to a COPY of the features, and is NEVER written back to `feature_matrix`
(storage always holds clean values). M1a wires the call site so the training
flow is final; the real per-bucket Gaussian injection (sized from
`config.features.weather.noise_sigma_by_bucket`) lands in M1b alongside the full
ensemble, where the hindcast-row sigma-selection design is resolved.

The stub returns `X` unchanged. M1b replaces the body WITHOUT touching the
assembly/training flow that calls it.
"""

from __future__ import annotations

import pandas as pd

from tennis.core.config import AppConfig


def apply_noise(X: pd.DataFrame, config: AppConfig) -> pd.DataFrame:  # noqa: ARG001
    """Return the training feature frame with forecast noise applied.

    M1a: no-op passthrough (returns `X` unchanged). The caller already passes a
    copy, so callers must not rely on identity. `config` is accepted now so the
    M1b implementation needs no signature change.
    """
    return X
