"""Shared §M15 forecast-uncertainty band logic.

Lives in `core` so BOTH the Research `conditions` extractor (which buckets a
forecast's `forecast_horizon_h`) AND the Modeling §M22 noise injector (which
buckets `decision_timing.live_decision_offset_hours`) resolve the band the SAME
way — there is a single owner of the locked half-open-lower-bands / clamped-top
convention, so the two consumers cannot drift. Models must not import from
agents, so the helper cannot live in `conditions.py`.
"""

from __future__ import annotations

from tennis.core.config import AppConfig
from tennis.core.errors import FeatureContractError

# One ordered band: (name, lo, hi). `hi` is the half-open upper bound for every
# band except the last, which clamps (matches any horizon >= its lo).
Band = tuple[str, int, int]


def build_uncertainty_bands(config: AppConfig) -> tuple[Band, ...]:
    """Order the configured uncertainty buckets into validated `(name, lo, hi)`
    triples.

    Raises `FeatureContractError` at construction (fail-fast, §M5/H2HConfig
    precedent) when the config is malformed, because `bucket_for_horizon` relies
    on the bands being ascending and non-overlapping — a config typo or reordering
    would otherwise SILENTLY mis-bucket forecasts (e.g. reversed buckets clamp a
    far-horizon forecast to "low"), which then mis-drives modeling-time
    noise-injection sigma with no loud failure (Codex R5b, MEDIUM). Validates:
      - each named bucket has a `[lo, hi]` threshold pair;
      - each band is non-empty (`lo < hi`);
      - bands are ascending and non-overlapping (`band[i].lo >= band[i-1].hi`).
    Gaps are NOT rejected — a horizon in a gap buckets to NULL (a safe absence,
    not a wrong value) and the real v1 config is contiguous."""
    thresholds = config.features.weather.uncertainty_bucket_thresholds
    bands: list[Band] = []
    for name in config.features.weather.uncertainty_buckets:
        pair = thresholds.get(name)
        if pair is None or len(pair) != 2:
            raise FeatureContractError(
                "weather.uncertainty_bucket_thresholds is missing a valid "
                f"[lo, hi] pair for bucket {name!r}: {thresholds!r}"
            )
        lo, hi = int(pair[0]), int(pair[1])
        if lo >= hi:
            raise FeatureContractError(
                f"weather uncertainty bucket {name!r} has a non-positive range "
                f"[lo={lo}, hi={hi}]; require lo < hi"
            )
        if bands and lo < bands[-1][2]:
            raise FeatureContractError(
                f"weather uncertainty buckets must be ascending and "
                f"non-overlapping: bucket {name!r} lo={lo} overlaps the previous "
                f"band {bands[-1][0]!r} hi={bands[-1][2]}"
            )
        bands.append((name, lo, hi))
    if not bands:
        raise FeatureContractError(
            "weather.uncertainty_buckets is empty; cannot bucket forecast horizons"
        )
    return tuple(bands)


def bucket_for_horizon(horizon_h: int, bands: tuple[Band, ...]) -> str | None:
    """The uncertainty bucket name for a forecast `horizon_h` (§M15).

    Half-open `[lo, hi)` for every band except the last, which clamps (matches any
    `horizon_h >= lo`) so a max-horizon forecast is bucketed, not dropped. Returns
    None for a horizon below the lowest band (defensive; should not occur)."""
    last = len(bands) - 1
    for i, (name, lo, hi) in enumerate(bands):
        if lo <= horizon_h < hi or (i == last and horizon_h >= lo):
            return name
    return None
