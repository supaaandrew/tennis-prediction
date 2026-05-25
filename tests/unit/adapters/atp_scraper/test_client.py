"""Unit tests for HttpAtpScraperClient. HTTP transport is faked; no real sleeps."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from tennis.adapters.atp_scraper.client import HttpAtpScraperClient
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import HttpResponse
from tennis.core.errors import AdapterError, RateLimitError, UpstreamUnavailableError


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    return load_config(config_path)


class _FakeHttp:
    """Returns queued responses; records every request (url + headers)."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(
        self, url: str, *, params: Any = None, headers: Any = None,
        timeout_s: float | None = None,
    ) -> HttpResponse:
        self.calls.append({"url": url, "headers": dict(headers or {})})
        if not self._responses:
            raise AssertionError("unexpected extra HTTP call")
        return self._responses.pop(0)

    def post(self, *args: Any, **kwargs: Any) -> HttpResponse:  # pragma: no cover
        raise NotImplementedError


class _RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _resp(status: int, body: bytes = b"<html></html>") -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body)


def _fast_monotonic() -> Any:
    """Monotonic that jumps far ahead each call so the throttle never sleeps."""
    counter = itertools.count(0, 1000)
    return lambda: float(next(counter))


def _client(
    config: AppConfig, http: _FakeHttp, sleep: _RecordingSleep
) -> HttpAtpScraperClient:
    return HttpAtpScraperClient(
        config=config, http=http, sleep=sleep, monotonic=_fast_monotonic(),
    )


# ---------------------------------------------------------------------------
# Success + URL shape
# ---------------------------------------------------------------------------
def test_index_returns_decoded_html(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b"<html><body>index</body></html>")])
    client = _client(config, http, _RecordingSleep())

    result = client.fetch_tournament_index()

    assert "index" in result
    assert isinstance(result, str)


def test_matches_url_includes_slug(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b"<html></html>")])
    client = _client(config, http, _RecordingSleep())

    client.fetch_tournament_matches("wimbledon")

    assert "wimbledon" in http.calls[0]["url"]


def test_non_strict_utf8_body_decodes_without_raising(config: AppConfig) -> None:
    # b"\xe9" is a valid latin-1 'é' but invalid standalone UTF-8.
    http = _FakeHttp([_resp(200, b"Caf\xe9 Centre")])
    client = _client(config, http, _RecordingSleep())

    result = client.fetch_tournament_index()

    assert isinstance(result, str)
    assert "Caf" in result  # errors='replace' kept the rest of the page


# ---------------------------------------------------------------------------
# User-agent rotation
# ---------------------------------------------------------------------------
def test_user_agent_rotates_deterministically(config: AppConfig) -> None:
    agents = config.sources.atp_scraper.user_agents
    assert len(agents) >= 2  # config fixture must have a rotation pool
    http = _FakeHttp([_resp(200) for _ in agents])
    client = _client(config, http, _RecordingSleep())

    for _ in agents:
        client.fetch_tournament_index()

    sent = [c["headers"]["User-Agent"] for c in http.calls]
    assert sent == list(agents)  # cycles through the pool in order


def test_user_agent_static_when_rotation_disabled(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200), _resp(200)])
    client = _client(config, http, _RecordingSleep())
    client._cfg = client._cfg.model_copy(update={"rotate_user_agent": False})

    client.fetch_tournament_index()
    client.fetch_tournament_index()

    sent = {c["headers"]["User-Agent"] for c in http.calls}
    assert sent == {config.sources.atp_scraper.user_agent}


def test_no_api_key_header_or_param(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200)])
    client = _client(config, http, _RecordingSleep())

    client.fetch_tournament_index()

    # Unauthenticated scraper: only the UA header, no auth material.
    assert set(http.calls[0]["headers"]) == {"User-Agent"}


# ---------------------------------------------------------------------------
# 429 — rate limited
# ---------------------------------------------------------------------------
def test_429_retries_with_backoff_then_raises(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429) for _ in range(4)])  # 1 initial + 3 retries
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(RateLimitError):
        client.fetch_tournament_index()

    assert len(http.calls) == 4
    assert len(sleep.calls) == 3
    assert sleep.calls[1] > sleep.calls[0]  # exponential


def test_429_then_200_succeeds(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429), _resp(200, b"<html>ok</html>")])
    client = _client(config, http, _RecordingSleep())

    assert "ok" in client.fetch_tournament_index()


# ---------------------------------------------------------------------------
# 403 — anti-bot block
# ---------------------------------------------------------------------------
def test_403_retries_rotating_ua_then_raises_upstream_unavailable(
    config: AppConfig,
) -> None:
    http = _FakeHttp([_resp(403) for _ in range(4)])
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(UpstreamUnavailableError):
        client.fetch_tournament_index()

    assert len(http.calls) == 4
    # Each retry rotated the UA — the first three differ across the pool.
    uas = [c["headers"]["User-Agent"] for c in http.calls]
    assert len(set(uas[: len(config.sources.atp_scraper.user_agents)])) > 1


def test_403_then_200_recovers(config: AppConfig) -> None:
    http = _FakeHttp([_resp(403), _resp(200, b"<html>recovered</html>")])
    client = _client(config, http, _RecordingSleep())

    assert "recovered" in client.fetch_tournament_index()


# ---------------------------------------------------------------------------
# 401 + other non-200
# ---------------------------------------------------------------------------
def test_401_raises_adapter_error_without_retry(config: AppConfig) -> None:
    http = _FakeHttp([_resp(401)])
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(AdapterError):
        client.fetch_tournament_index()

    assert len(http.calls) == 1
    assert sleep.calls == []


def test_503_raises_upstream_unavailable(config: AppConfig) -> None:
    http = _FakeHttp([_resp(503)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(UpstreamUnavailableError):
        client.fetch_tournament_index()


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------
def test_throttle_waits_between_successive_requests(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200), _resp(200)])
    sleep = _RecordingSleep()
    client = HttpAtpScraperClient(
        config=config, http=http, sleep=sleep, monotonic=lambda: 0.0,
    )

    client.fetch_tournament_index()
    client.fetch_tournament_index()

    expected = 1.0 / config.sources.atp_scraper.rate_limit_rps
    assert sleep.calls == [pytest.approx(expected)]
