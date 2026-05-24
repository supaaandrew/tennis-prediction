"""Unit tests for HttpOWMClient. HTTP transport is faked; no real sleeps."""

from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from tennis.adapters.owm.client import HttpOWMClient
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import HttpResponse
from tennis.core.errors import AdapterError, RateLimitError, UpstreamUnavailableError

API_KEY = "super-secret-owm-key-9f8e7d"


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


def _client(config: AppConfig, http: _FakeHttp, sleep: _RecordingSleep) -> HttpOWMClient:
    return HttpOWMClient(
        config=config, http=http, api_key=API_KEY,
        sleep=sleep, monotonic=_fast_monotonic(),
    )


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------
def test_successful_response_returns_parsed_dict(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b'{"temperature": {"afternoon": 300.0}}')])
    client = _client(config, http, _RecordingSleep())

    result = client.fetch_day_summary(1.0, 2.0, date(2020, 1, 1))

    assert result == {"temperature": {"afternoon": 300.0}}


def test_request_includes_appid_and_date(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b"{}")])
    client = _client(config, http, _RecordingSleep())

    client.fetch_day_summary(10.5, -20.25, date(2021, 6, 15))

    params = http.calls[0]["params"]
    assert params["appid"] == API_KEY
    assert params["lat"] == 10.5
    assert params["lon"] == -20.25
    assert params["date"] == "2021-06-15"


# ---------------------------------------------------------------------------
# 429 — rate limited
# ---------------------------------------------------------------------------
def test_429_retries_with_backoff_then_raises(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429) for _ in range(4)])  # 1 initial + 3 retries
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(RateLimitError):
        client.fetch_forecast(1.0, 2.0)

    assert len(http.calls) == 4  # exhausted all attempts
    assert len(sleep.calls) == 3  # one backoff per retry
    assert sleep.calls == sorted(sleep.calls)  # exponential, non-decreasing
    assert sleep.calls[1] > sleep.calls[0]  # strictly growing


def test_429_then_200_succeeds(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429), _resp(200, b'{"ok": 1}')])
    client = _client(config, http, _RecordingSleep())

    assert client.fetch_forecast(1.0, 2.0) == {"ok": 1}


# ---------------------------------------------------------------------------
# 401 — bad key (no retry)
# ---------------------------------------------------------------------------
def test_401_raises_adapter_error_without_retry(config: AppConfig) -> None:
    http = _FakeHttp([_resp(401)])
    sleep = _RecordingSleep()
    client = _client(config, http, sleep)

    with pytest.raises(AdapterError):
        client.fetch_forecast(1.0, 2.0)

    assert len(http.calls) == 1  # no retry
    assert sleep.calls == []  # no backoff


# ---------------------------------------------------------------------------
# Other non-200
# ---------------------------------------------------------------------------
def test_500_raises_upstream_unavailable(config: AppConfig) -> None:
    http = _FakeHttp([_resp(503)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(UpstreamUnavailableError):
        client.fetch_forecast(1.0, 2.0)


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------
def test_api_key_never_appears_in_logs(config: AppConfig) -> None:
    # 429 then 200 forces the "owm_rate_limited" warning to fire.
    http = _FakeHttp([_resp(429), _resp(200, b"{}")])
    client = _client(config, http, _RecordingSleep())

    with capture_logs() as logs:
        client.fetch_forecast(1.0, 2.0)

    assert any(entry.get("event") == "owm_rate_limited" for entry in logs)
    assert API_KEY not in repr(logs)


# ---------------------------------------------------------------------------
# Secret hygiene on the error paths — exception messages must NEVER carry
# the API key (or the URL/params dict that contains it). These tests are the
# regression guard for the `appid` param assembled in `_get_json`.
# ---------------------------------------------------------------------------
def test_adapter_error_message_does_not_leak_api_key(config: AppConfig) -> None:
    http = _FakeHttp([_resp(401)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(AdapterError) as exc_info:
        client.fetch_forecast(1.0, 2.0)

    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(exc_info.value)


def test_rate_limit_error_message_does_not_leak_api_key(config: AppConfig) -> None:
    http = _FakeHttp([_resp(429) for _ in range(4)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(RateLimitError) as exc_info:
        client.fetch_forecast(1.0, 2.0)

    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(exc_info.value)


def test_upstream_unavailable_error_message_does_not_leak_api_key(
    config: AppConfig,
) -> None:
    http = _FakeHttp([_resp(503)])
    client = _client(config, http, _RecordingSleep())

    with pytest.raises(UpstreamUnavailableError) as exc_info:
        client.fetch_forecast(1.0, 2.0)

    assert API_KEY not in str(exc_info.value)
    assert API_KEY not in repr(exc_info.value)


# ---------------------------------------------------------------------------
# Rate-limit throttle
# ---------------------------------------------------------------------------
def test_throttle_waits_between_successive_requests(config: AppConfig) -> None:
    http = _FakeHttp([_resp(200, b"{}"), _resp(200, b"{}")])
    sleep = _RecordingSleep()
    # Constant monotonic => second request sees zero elapsed => must wait.
    client = HttpOWMClient(
        config=config, http=http, api_key=API_KEY,
        sleep=sleep, monotonic=lambda: 0.0,
    )

    client.fetch_forecast(1.0, 2.0)  # first call: no wait
    client.fetch_forecast(1.0, 2.0)  # second call: throttled

    expected = 1.0 / config.sources.openweather.rate_limit_rps
    assert sleep.calls == [pytest.approx(expected)]
