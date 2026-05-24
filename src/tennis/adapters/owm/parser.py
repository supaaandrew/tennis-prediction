"""Pure parsers for OpenWeatherMap One Call 3.0 responses.

Zero side effects: every function takes a response dict (or one hourly
entry) plus explicit parameters and returns a `WeatherObservationRow`.
No HTTP, no database, no config object. Missing fields collapse to None
rather than crashing — a sparse upstream payload still produces a row.

Two endpoints map to two parsers (DECISIONS.md §15.3, H4, H5):
  - ``day_summary``    -> ``parse_day_summary``   (historical hindcast)
  - ``onecall.hourly`` -> ``parse_forecast_hour`` (forecast, horizon-gated)

``uncertainty_bucket`` exists ONLY to decide skip-vs-keep on a forecast
hour: a horizon beyond the configured ``high`` band is dropped. The bucket
value itself is never persisted — ``feature_matrix`` stores clean values
and the Modeling Agent derives the noise bucket at train time (H1, §15.5).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from tennis.core.logging import get_logger
from tennis.storage.postgres.rows import WeatherObservationRow

_logger = get_logger("tennis.adapters.owm.parser")

_SOURCE = "owm"
_KELVIN_OFFSET = 273.15


def _require_aware(dt: datetime, *, label: str) -> None:
    """Raise `ValueError` unless `dt` is timezone-aware (UTC-only contract)."""
    if dt.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")


def _kelvin_to_celsius(kelvin: float | None) -> float | None:
    """Convert Kelvin to Celsius; None passes through untouched."""
    if kelvin is None:
        return None
    return kelvin - _KELVIN_OFFSET


def _coerce_precip(value: Any) -> float:
    """Return a non-negative float for a precipitation field.

    Defensive against malformed upstream payloads: non-numeric values (strings,
    bools, dicts) collapse to 0.0; numeric negatives clamp to 0.0 — negative
    precipitation is physically invalid and would silently poison feature stats.
    """
    if value is None or isinstance(value, bool):
        # bool is a subclass of int; treat True/False as garbage, not 1/0.
        return 0.0
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return n if n > 0.0 else 0.0


def _nested(data: Any, *keys: str) -> Any:
    """Return ``data[k1][k2]...`` or None if any level is missing.

    Tolerates a non-mapping at any depth (returns None) so a malformed
    upstream payload yields NULL columns instead of raising.
    """
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def uncertainty_bucket(
    horizon_h: int, thresholds: Mapping[str, list[int]]
) -> str | None:
    """Map a forecast horizon (hours) to its uncertainty band (H4).

    Boundaries are lower-inclusive / upper-exclusive so each hour lands in
    exactly one band; the final ``high`` upper bound is inclusive. A horizon
    beyond ``high`` (or negative) returns None, signalling the caller to skip
    the row — it is past ``max_forecast_horizon_h``.

    With config bands low=[0,6], medium=[6,24], high=[24,168]:
    ``0->low, 6->medium, 24->high, 168->high, 169->None``.
    """
    low = thresholds.get("low")
    medium = thresholds.get("medium")
    high = thresholds.get("high")
    if low and low[0] <= horizon_h < low[1]:
        return "low"
    if medium and medium[0] <= horizon_h < medium[1]:
        return "medium"
    if high and high[0] <= horizon_h <= high[1]:
        return "high"
    return None


def parse_day_summary(
    response: Mapping[str, Any], venue_id: int, observed_at: datetime
) -> WeatherObservationRow:
    """Parse a One Call 3.0 ``day_summary`` response into a hindcast row.

    `observed_at` is supplied by the adapter (the canonical UTC instant for
    the summarised day). Field paths follow DECISIONS.md §15.3; cloud cover
    is absent from ``day_summary`` so `cloud_pct` is always None, and the
    forecast fields mark this row as a hindcast (`is_forecast=False`,
    `forecast_horizon_h=None`).
    """
    _require_aware(observed_at, label="observed_at")
    return WeatherObservationRow(
        venue_id=venue_id,
        observed_at=observed_at,
        source=_SOURCE,
        is_forecast=False,
        temp_c=_kelvin_to_celsius(_nested(response, "temperature", "afternoon")),
        humidity_pct=_nested(response, "humidity", "afternoon"),
        wind_speed_ms=_nested(response, "wind", "max", "speed"),
        wind_dir_deg=_nested(response, "wind", "max", "direction"),
        pressure_hpa=_nested(response, "pressure", "afternoon"),
        precip_mm=_nested(response, "precipitation", "total") or 0.0,
        cloud_pct=None,
        forecast_horizon_h=None,
    )


def parse_forecast_hour(
    hour: Mapping[str, Any],
    venue_id: int,
    fetch_ts: datetime,
    *,
    thresholds: Mapping[str, list[int]],
) -> WeatherObservationRow | None:
    """Parse one ``onecall`` hourly entry into a forecast row, or None to skip.

    `observed_at` comes from the entry's unix ``dt``; `forecast_horizon_h` is
    the rounded hour gap from `fetch_ts`. When the horizon falls outside the
    configured bands (beyond ``max_forecast_horizon_h``, or in the past) the
    entry is skipped (returns None). `precip_mm` sums rain+snow ``1h`` totals
    per DECISIONS.md §15.3.
    """
    _require_aware(fetch_ts, label="fetch_ts")
    dt_unix = hour.get("dt")
    if dt_unix is None:
        return None
    observed_at = datetime.fromtimestamp(dt_unix, tz=UTC)
    horizon_h = round((observed_at - fetch_ts).total_seconds() / 3600)
    if uncertainty_bucket(horizon_h, thresholds) is None:
        return None

    # `_coerce_precip` collapses non-numeric / None / negative to 0.0; the
    # outer `max(0.0, ...)` is a belt-and-suspenders clamp in case both inputs
    # are numerically degenerate (e.g. -inf summed with NaN-equivalent).
    rain = _coerce_precip(_nested(hour, "rain", "1h"))
    snow = _coerce_precip(_nested(hour, "snow", "1h"))
    precip_mm = max(0.0, rain + snow)
    return WeatherObservationRow(
        venue_id=venue_id,
        observed_at=observed_at,
        source=_SOURCE,
        is_forecast=True,
        temp_c=_kelvin_to_celsius(hour.get("temp")),
        humidity_pct=hour.get("humidity"),
        wind_speed_ms=hour.get("wind_speed"),
        wind_dir_deg=hour.get("wind_deg"),
        pressure_hpa=hour.get("pressure"),
        precip_mm=precip_mm,
        cloud_pct=hour.get("clouds"),
        forecast_horizon_h=horizon_h,
    )
