"""HTTP client for The Odds API v4.

Pure transport: builds requests, enforces the per-source rate limit, and
maps HTTP status codes to the typed adapter errors. No parsing — callers
get the decoded JSON (a list of events for current odds, or the historical
snapshot wrapper) and hand it to `parser`.

Status handling (identical to the OWM client, §O):
  200  -> decoded JSON
  401  -> `AdapterError` immediately (bad API key; retrying cannot help)
  429  -> exponential backoff, up to ``_MAX_RETRIES``, then `RateLimitError`
  else -> `UpstreamUnavailableError`

The API key is sent as the ``apiKey`` query param and is NEVER logged — not
in a log kwarg, not in a raised message, not in a dead-letter payload; the
structlog redactor is only a backstop. `sleep` and `monotonic` are injectable
so tests exercise throttle/backoff without real wall-clock delays.

``sport`` and ``bookmakers`` are read from ``config.sources.odds_api`` (the
client owns the source's knobs, mirroring how the OWM client owns its base
URL and endpoints); only the per-call variable — the historical snapshot
timestamp — is passed in by the adapter.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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

_MARKET_H2H = "h2h"       # only h2h ingested for v1 (§15.4)
_ODDS_FORMAT = "decimal"


@runtime_checkable
class OddsApiClient(Protocol):
    """Read side of The Odds API. Injected so the adapter can be unit-tested
    without real HTTP."""

    def fetch_current_odds(self) -> list[dict[str, Any]]: ...
    def fetch_historical_odds(self, snapshot_iso: str) -> dict[str, Any]: ...


class HttpOddsApiClient:
    """Concrete `OddsApiClient` over the shared `HttpClient` transport."""

    def __init__(
        self,
        *,
        config: AppConfig,
        http: HttpClient,
        api_key: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config.sources.odds_api
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
        self._logger = get_logger("tennis.adapters.odds.client")

    # -- public API ---------------------------------------------------------
    def fetch_current_odds(self) -> list[dict[str, Any]]:
        """Upcoming/live events with odds. Returns a list of event dicts."""
        url = f"{self._cfg.base_url}/sports/{self._cfg.sport}/odds"
        return self._request(url, self._query())

    def fetch_historical_odds(self, snapshot_iso: str) -> dict[str, Any]:
        """Odds as they stood at ``snapshot_iso``. Returns the historical
        wrapper: ``{timestamp, previous_timestamp, next_timestamp, data}``."""
        url = f"{self._cfg.base_url}/historical/sports/{self._cfg.sport}/odds"
        return self._request(url, self._query(date=snapshot_iso))

    # -- transport ----------------------------------------------------------
    def _query(self, **extra: Any) -> dict[str, Any]:
        """Assemble query params. ``apiKey`` lives only here — never logged."""
        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "bookmakers": ",".join(self._cfg.bookmakers),
            "markets": _MARKET_H2H,
            "oddsFormat": _ODDS_FORMAT,
        }
        params.update(extra)
        return params

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

    def _request(self, url: str, params: dict[str, Any]) -> Any:
        backoff = self._backoff_base_s
        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            response = self._http.get(url, params=params, timeout_s=_REQUEST_TIMEOUT_S)
            status = response.status
            if status == _HTTP_OK:
                return response.json()
            if status == _HTTP_UNAUTHORIZED:
                # Bad key — retrying cannot help. Do NOT log or echo the key.
                raise AdapterError("The Odds API rejected the API key (HTTP 401)")
            if status == _HTTP_RATE_LIMITED:
                if attempt < _MAX_RETRIES:
                    self._logger.warning(
                        "odds_rate_limited", attempt=attempt + 1, backoff_s=backoff
                    )
                    self._sleep(backoff)
                    backoff *= 2
                    continue
                raise RateLimitError(
                    f"The Odds API rate limit exceeded after {_MAX_RETRIES} retries"
                )
            raise UpstreamUnavailableError(f"The Odds API returned HTTP {status}")
        # The loop always returns or raises; this guards against a logic slip.
        raise UpstreamUnavailableError(  # pragma: no cover
            "The Odds API request loop exhausted without a response"
        )
