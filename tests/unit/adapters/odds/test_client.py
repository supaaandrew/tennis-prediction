"""Unit tests for HttpOddsApiClient. HTTP transport is faked; no real sleeps."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from tennis.adapters.odds.client import HttpOddsApiClient
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import HttpResponse
from tennis.core.errors import AdapterError, RateLimitError, UpstreamUnavailableError

API_KEY = "super-secret-odds-key-9f8e7d"


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    return load_config(config_path)


class _FakeHttp:
    """Returns queued responses; records every request for assertions."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(
        self, url: str, *, params: Any = None, headers: Any = None,
        timeout_s: float | None = None,
    ) -> HttpResponse:
        self.calls.append({"url": url, "params": dict(params or {})})
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


def _resp(status: int, body: bytes = b"{}") -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body)


def _fast_monotonic() -> Any:
    """Monotonic that jumps far ahead each call so the throttle never sleeps."""
    counter = itertools.count(0, 1000)
    return lambda: float(next(counter))


def _client(
    config: AppConfig, http: _FakeHttp, sleep: _RecordingSleep
) -> HttpOddsApiClient:
    return HttpOddsApiClient(
        config=config, http=http, api_key=API_KEY,
        sleep=sleep, monotonic=_fast_monotonic(),
    )


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------
def test_current_odds_returns_parsed_list(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b'[{"id": "abc"}]')])
    client = _client(config, http, _RecordingSleep())

    result = client.fetch_current_odds()

    assert result == [{"id": "abc"}]


def test_historical_odds_returns_wrapper_dict(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b'{"timestamp": "2020-06-01T00:00:00Z", "data": []}')])
    client = _client(config, http, _RecordingSleep())

    result = client.fetch_historical_odds("2020-06-01T00:00:00Z")

    assert result["timestamp"] == "2020-06-01T00:00:00Z"
    assert result["data"] == []


def test_request_includes_apikey_bookmakers_and_market(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b"[]")])
    client = _client(config, http, _RecordingSleep())

    client.fetch_current_odds()

    params = http.calls[0]["params"]
    assert params["apiKey"] == API_KEY
    assert params["markets"] == "h2h"
    assert params["oddsFormat"] == "decimal"
    # bookmakers comes straight from config — comma joined.
    for book in config.sources.odds_api.bookmakers:
        assert book in params["bookmakers"]


def test_historical_request_includes_date_param(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b'{"data": []}')])
    client = _client(config, http, _RecordingSleep())

    client.fetch_historical_odds("2021-06-15T12:00:00Z")

    assert http.calls[0]["params"]["date"] == "2021-06-15T12:00:00Z"
    assert "/historical/" in http.calls[0]["url"]


# ---------------------------------------------------------------------------
# 429 — rate limited
# ---------------------------------------------------------------------------
def test_429_retries_with_backoff_then_raises(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429) for _ in range(4)])  # 1 initial + 3 retries
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(RateLimitError):
        client.fetch_current_odds()

    assert len(http.calls) == 4  # exhausted all attempts
    assert len(sleep.calls) == 3  # one backoff per retry
    assert sleep.calls == sorted(sleep.calls)  # exponential, non-decreasing
    assert sleep.calls[1] > sleep.calls[0]  # strictly growing


def test_429_then_200_succeeds(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429), _resp(200, b'[{"ok": 1}]')])
    client = _client(config, http, _RecordingSleep())

    assert client.fetch_current_odds() == [{"ok": 1}]


# ---------------------------------------------------------------------------
# 401 — bad key (no retry)
# ---------------------------------------------------------------------------
def test_401_raises_adapter_error_without_retry(config: AppConfig) -> None:
    http = _FakeHttp([_resp(401)])
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(AdapterError):
        client.fetch_current_odds()

    assert len(http.calls) == 1  # no retry
    assert sleep.calls == []  # no backoff


# ---------------------------------------------------------------------------
# Other non-200
# ---------------------------------------------------------------------------
def test_503_raises_upstream_unavailable(config: AppConfig) -> None:
    http = _FakeHttp([_resp(503)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(UpstreamUnavailableError):
        client.fetch_current_odds()


# ---------------------------------------------------------------------------
# Secret hygiene — the API key must never surface in logs or exceptions.
# ---------------------------------------------------------------------------
def test_api_key_never_appears_in_logs(config: AppConfig) -> None:
    # 429 then 200 forces the "odds_rate_limited" warning to fire.
    http = _FakeHttp([_resp(429), _resp(200, b"[]")])
    client = _client(config, http, _RecordingSleep())

    with capture_logs() as logs:
        client.fetch_current_odds()

    assert any(entry.get("event") == "odds_rate_limited" for entry in logs)
    assert API_KEY not in repr(logs)


def test_adapter_error_message_does_not_leak_api_key(config: AppConfig) -> None:
    http = _FakeHttp([_resp(401)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(AdapterError) as exc_info:
        client.fetch_current_odds()

    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(exc_info.value)


def test_rate_limit_error_message_does_not_leak_api_key(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429) for _ in range(4)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(RateLimitError) as exc_info:
        client.fetch_current_odds()

    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(exc_info.value)


def test_upstream_unavailable_error_message_does_not_leak_api_key(
    config: AppConfig,
) -> None:
    http = _FakeHttp([_resp(503)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(UpstreamUnavailableError) as exc_info:
        client.fetch_current_odds()

    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(exc_info.value)


# ---------------------------------------------------------------------------
# Rate-limit throttle
# ---------------------------------------------------------------------------
def test_throttle_waits_between_successive_requests(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b"[]"), _resp(200, b"[]")])
    sleep = _RecordingSleep()
    # Constant monotonic => second request sees zero elapsed => must wait.
    client = HttpOddsApiClient(
        config=config, http=http, api_key=API_KEY,
        sleep=sleep, monotonic=lambda: 0.0,
    )

    client.fetch_current_odds()  # first call: no wait
    client.fetch_current_odds()  # second call: throttled

    expected = 1.0 / config.sources.odds_api.rate_limit_rps
    assert sleep.calls == [pytest.approx(expected)]
