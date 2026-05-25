"""HTTP client for the ATP website scraper (`atptour.com`).

Pure transport: builds requests, rotates the User-Agent header, enforces the
per-source rate limit, and maps HTTP status codes to the typed adapter errors.
No parsing — callers get raw HTML (`str`) and hand it to `parser`.

The scraper is **unauthenticated** — there is no API key anywhere. Status
handling mirrors the OWM/Odds clients (§O / §J) with one addition for
anti-bot 403s:
  200  -> decoded HTML string
  401  -> `AdapterError` immediately (no retry can help)
  429  -> exponential backoff, up to ``_MAX_RETRIES``, then `RateLimitError`
  403  -> exponential backoff rotating the UA, up to ``_MAX_RETRIES``, then
          `UpstreamUnavailableError` (persistent anti-bot block; no headless
          browser in v1 — the adapter dead-letters the page)
  else -> `UpstreamUnavailableError`

`sleep` and `monotonic` are injectable so tests exercise throttle/backoff
without real wall-clock delays; UA rotation uses a deterministic internal
counter so the chosen header sequence is reproducible in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from tennis.core.config import AppConfig
from tennis.core.contracts import HttpClient
from tennis.core.errors import AdapterError, RateLimitError, UpstreamUnavailableError
from tennis.core.logging import get_logger

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_RATE_LIMITED = 429
_MAX_RETRIES = 3  # 429/403 retries before giving up


@runtime_checkable
class AtpScraperClient(Protocol):
    """Read side of the ATP website. Injected so the adapter can be
    unit-tested without real HTTP."""

    def fetch_tournament_index(self) -> str: ...
    def fetch_tournament_matches(self, slug: str) -> str: ...


class HttpAtpScraperClient:
    """Concrete `AtpScraperClient` over the shared `HttpClient` transport."""

    def __init__(
        self,
        *,
        config: AppConfig,
        http: HttpClient,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config.sources.atp_scraper
        self._http = http
        self._sleep = sleep
        self._monotonic = monotonic
        self._backoff_base_s = config.ingestion.retry.base_delay_s
        self._timeout_s = float(self._cfg.request_timeout_s)
        rps = self._cfg.rate_limit_rps
        self._min_interval_s = (1.0 / rps) if rps > 0 else 0.0
        self._last_request_at: float | None = None
        self._ua_count = 0
        self._logger = get_logger("tennis.adapters.atp_scraper.client")

    # -- public API ---------------------------------------------------------
    def fetch_tournament_index(self) -> str:
        """HTML of the current/recent tournaments index page."""
        return self._request(f"{self._cfg.base_url}/en/scores/current")

    def fetch_tournament_matches(self, slug: str) -> str:
        """HTML of one tournament's results/schedule page."""
        return self._request(f"{self._cfg.base_url}/en/scores/current/{slug}/results")

    # -- transport ----------------------------------------------------------
    def _next_user_agent(self) -> str:
        """Pick the next UA. Rotates deterministically through ``user_agents``
        when configured; otherwise the single ``user_agent``."""
        agents = self._cfg.user_agents
        if self._cfg.rotate_user_agent and agents:
            ua = agents[self._ua_count % len(agents)]
            self._ua_count += 1
            return ua
        return self._cfg.user_agent

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

    def _request(self, url: str) -> str:
        backoff = self._backoff_base_s
        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            user_agent = self._next_user_agent()
            response = self._http.get(
                url, headers={"User-Agent": user_agent}, timeout_s=self._timeout_s
            )
            status = response.status
            if status == _HTTP_OK:
                # ATP pages are not always strict UTF-8; replace rather than
                # raise so a stray byte never aborts the run in transport.
                return response.body.decode("utf-8", errors="replace")
            if status == _HTTP_UNAUTHORIZED:
                raise AdapterError("ATP website rejected the request (HTTP 401)")
            if status in (_HTTP_RATE_LIMITED, _HTTP_FORBIDDEN):
                if attempt < _MAX_RETRIES:
                    self._logger.warning(
                        "atp_scraper_retry",
                        status=status,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                    )
                    self._sleep(backoff)
                    backoff *= 2
                    continue
                if status == _HTTP_RATE_LIMITED:
                    raise RateLimitError(
                        f"ATP website rate limit exceeded after {_MAX_RETRIES} retries"
                    )
                # Persistent 403 despite UA rotation — anti-bot block.
                raise UpstreamUnavailableError(
                    f"ATP website returned HTTP 403 after {_MAX_RETRIES} retries "
                    "(anti-bot; headless browser deferred to a future extension)"
                )
            raise UpstreamUnavailableError(f"ATP website returned HTTP {status}")
        raise UpstreamUnavailableError(  # pragma: no cover
            "ATP website request loop exhausted without a response"
        )
