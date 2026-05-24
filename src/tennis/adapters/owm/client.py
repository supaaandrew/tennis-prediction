"""HTTP client for the OpenWeatherMap One Call 3.0 API.

Pure transport: builds requests, enforces the per-source rate limit, and
maps HTTP status codes to the typed adapter errors. No parsing — callers
get the decoded JSON dict and hand it to `parser`.

Status handling:
  200  -> decoded JSON
  401  -> `AdapterError` immediately (bad API key; retrying cannot help)
  429  -> exponential backoff, up to ``_MAX_RETRIES``, then `RateLimitError`
  else -> `UpstreamUnavailableError`

The API key is sent as the ``appid`` query param and is NEVER logged — not
even via a log kwarg; the structlog redactor is only a backstop. `sleep` and
`monotonic` are injectable so tests exercise throttle/backoff without real
wall-clock delays.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol, runtime_checkable

from tennis.core.config import AppConfig, read_required_env
from tennis.core.contracts import HttpClient
from tennis.core.errors import AdapterError, RateLimitError, UpstreamUnavailableError
from tennis.core.logging import get_logger

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_HTTP_RATE_LIMITED = 429
_MAX_RETRIES = 3          # 429 retries before giving up
_REQUEST_TIMEOUT_S = 30.0


@runtime_checkable
class OWMClient(Protocol):
    """Read side of the OpenWeatherMap API. Injected so the adapter can be
    unit-tested without real HTTP."""

    def fetch_day_summary(self, lat: float, lon: float, day: date) -> dict[str, Any]: ...
    def fetch_forecast(self, lat: float, lon: float) -> dict[str, Any]: ...


class HttpOWMClient:
    """Concrete `OWMClient` over the shared `HttpClient` transport."""

    def __init__(
        self,
        *,
        config: AppConfig,
        http: HttpClient,
        api_key: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config.sources.openweather
        self._http = http
        # Resolve lazily-but-eagerly here so a missing key fails at adapter
        # construction, not mid-backfill. Never stored in a logged structure.
        self._api_key = api_key or read_required_env(self._cfg.api_key_env)
        self._sleep = sleep
        self._monotonic = monotonic
        self._backoff_base_s = config.ingestion.retry.base_delay_s
        rps = self._cfg.rate_limit_rps
        self._min_interval_s = (1.0 / rps) if rps > 0 else 0.0
        self._last_request_at: float | None = None
        self._logger = get_logger("tennis.adapters.owm.client")

    # -- public API ---------------------------------------------------------
    def fetch_day_summary(self, lat: float, lon: float, day: date) -> dict[str, Any]:
        url = f"{self._cfg.base_url}{self._cfg.historical_endpoint}"
        return self._get_json(url, {"lat": lat, "lon": lon, "date": day.isoformat()})

    def fetch_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        url = f"{self._cfg.base_url}{self._cfg.forecast_endpoint}"
        return self._get_json(url, {"lat": lat, "lon": lon})

    # -- transport ----------------------------------------------------------
    def _throttle(self) -> None:
        """Block as needed so successive requests honour ``rate_limit_rps``."""
        if self._min_interval_s <= 0:
            return
        if self._last_request_at is not None:
            elapsed = self._monotonic() - self._last_request_at
            wait = self._min_interval_s - elapsed
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._monotonic()

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {**params, "appid": self._api_key}
        backoff = self._backoff_base_s
        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            response = self._http.get(
                url, params=request_params, timeout_s=_REQUEST_TIMEOUT_S
            )
            status = response.status
            if status == _HTTP_OK:
                return response.json()
            if status == _HTTP_UNAUTHORIZED:
                # Bad key — retrying cannot help. Do NOT log the key.
                raise AdapterError("OpenWeatherMap rejected the API key (HTTP 401)")
            if status == _HTTP_RATE_LIMITED:
                if attempt < _MAX_RETRIES:
                    self._logger.warning(
                        "owm_rate_limited", attempt=attempt + 1, backoff_s=backoff
                    )
                    self._sleep(backoff)
                    backoff *= 2
                    continue
                raise RateLimitError(
                    f"OpenWeatherMap rate limit exceeded after {_MAX_RETRIES} retries"
                )
            raise UpstreamUnavailableError(f"OpenWeatherMap returned HTTP {status}")
        # The loop always returns or raises; this guards against a logic slip.
        raise UpstreamUnavailableError(  # pragma: no cover
            "OpenWeatherMap request loop exhausted without a response"
        )
