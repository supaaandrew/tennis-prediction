"""H1 forecast-noise injection (§M22).

H1: forecast-noise injection lives in the Modeling training loop ONLY, applied
to a COPY of the features, and is NEVER written back to `feature_matrix`
(storage always holds clean values). The prediction path NEVER calls this.

§M22 — sigma selection on hindcast training rows. Training rows are hindcast, so
`forecast_uncertainty_bucket` is NULL → there is no per-row sigma. The injection
sizes its noise from the single uncertainty bucket that corresponds to
`config.decision_timing.live_decision_offset_hours`, resolved through the same
§M15 half-open bands the `conditions` extractor uses (24h → "high" for the locked
v1 offset). Per-column sigmas come from
`config.features.weather.noise_sigma_by_bucket[bucket]`, keyed by the bare metric
name (`temp_c`/`humidity_pct`/`wind_speed_ms`); the matching feature column is
`<metric>_decision`. Columns with no matching sigma are left untouched.

The RNG is seeded from `config.modeling.random_seed` so each retrain — and the
unit suite — is reproducible. `inject_forecast_noise=false` is honored as a no-op
(logged as the §M21 weather hindcast/forecast-skew risk). NaN cells stay NaN
(noise is added; a missing value is never invented).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.core.weather_uncertainty import (
    build_uncertainty_bands,
    bucket_for_horizon,
)

_logger = get_logger("tennis.models.noise")

# §M22: a weather feature column is the bare sigma-metric name + this suffix.
_DECISION_SUFFIX = "_decision"


def apply_noise(X: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Return a COPY of the training feature frame with forecast noise applied.

    Training-only (H1). `inject_forecast_noise=false` → an unchanged copy. The
    returned frame is always a distinct object from the input, so the caller's
    `feature_matrix`-derived frame is never mutated.
    """
    out = X.copy()
    weather = config.features.weather
    if not weather.inject_forecast_noise:
        _logger.warning(
            "noise_injection_disabled",
            reason="config.features.weather.inject_forecast_noise=false",
            risk="weather hindcast/forecast skew not absorbed (H1/§M21)",
        )
        return out

    # §M22: pick the decision-offset bucket via the shared §M15 bands. _build_bands
    # fails loud (FeatureContractError) on malformed config, mirroring the
    # conditions extractor — a silent mis-bucket would mis-size the sigma.
    bands = build_uncertainty_bands(config)
    offset_h = config.decision_timing.live_decision_offset_hours
    bucket = bucket_for_horizon(offset_h, bands)
    sigma_by_metric = weather.noise_sigma_by_bucket.get(bucket) if bucket else None
    if not sigma_by_metric:
        _logger.warning(
            "noise_injection_no_sigma",
            offset_hours=offset_h,
            bucket=bucket,
            detail="no noise_sigma_by_bucket entry for the decision-offset bucket",
        )
        return out

    rng = np.random.default_rng(config.modeling.random_seed)
    applied: list[str] = []
    for metric, sigma in sigma_by_metric.items():
        col = f"{metric}{_DECISION_SUFFIX}"
        if col not in out.columns or sigma <= 0:
            continue
        noise = rng.normal(0.0, float(sigma), size=len(out))
        # Adding to a NaN cell yields NaN — native missing is preserved (§M20a).
        out[col] = out[col].to_numpy(dtype="float64") + noise
        applied.append(col)

    _logger.info(
        "noise_injected",
        bucket=bucket,
        offset_hours=offset_h,
        columns=applied,
        seed=config.modeling.random_seed,
    )
    return out
