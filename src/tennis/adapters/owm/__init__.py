"""OpenWeatherMap adapter (weather source).

Fetches weather from the OpenWeatherMap One Call 3.0 API and writes
`weather_observations` rows through the storage repositories. Two modes:

  - historical backfill via ``day_summary`` (DECISIONS.md H5 — one call per
    day vs. 24× for ``timemachine``)
  - hourly forecast via ``onecall``, gated by ``max_forecast_horizon_h`` (H4)

Layers mirror the Sackmann adapter: `client` (HTTP transport), `parser`
(pure response -> Row DTO), `adapter` (orchestration, watermark, dead-letter).
See DECISIONS.md §15.3 for the field → column mapping.
"""

from tennis.adapters.owm.adapter import (
    BackfillResult,
    ForecastResult,
    OWMAdapter,
)
from tennis.adapters.owm.client import HttpOWMClient, OWMClient
from tennis.adapters.owm.parser import (
    parse_day_summary,
    parse_forecast_hour,
    uncertainty_bucket,
)

__all__ = [
    # client
    "OWMClient",
    "HttpOWMClient",
    # parser
    "parse_day_summary",
    "parse_forecast_hour",
    "uncertainty_bucket",
    # adapter
    "OWMAdapter",
    "BackfillResult",
    "ForecastResult",
]
