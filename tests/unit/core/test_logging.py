"""Logging tests — JSON output, contextvars, secret redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

from tennis.core.logging import (
    bind_contextvars,
    clear_contextvars,
    configure_logging,
    get_logger,
)


def _last_json_line(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out.strip()
    assert out, "expected at least one log line"
    return json.loads(out.splitlines()[-1])


def _emit(call: Callable[..., Any], **kwargs: Any) -> None:
    call(**kwargs)
    for h in logging.getLogger().handlers:
        h.flush()


class TestConfigure:
    def test_json_output_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json=True)
        log = get_logger("tennis.test")
        _emit(log.info, event="hello", value=1)
        record = _last_json_line(capsys)
        assert record["event"] == "hello"
        assert record["value"] == 1
        assert record["level"] == "info"
        assert "timestamp" in record

    def test_safe_to_reconfigure(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="WARNING", json=True)
        configure_logging(level="INFO", json=True)
        log = get_logger()
        _emit(log.info, event="ok")
        record = _last_json_line(capsys)
        assert record["event"] == "ok"


class TestRedaction:
    def test_redacts_secret_key_at_top_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json=True, redact_keys=("api_key", "password"))
        log = get_logger()
        _emit(log.info, event="call", api_key="abcdef")
        record = _last_json_line(capsys)
        assert record["api_key"] == "***"
        assert record["event"] == "call"

    def test_redacts_substring_match(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json=True, redact_keys=("api_key",))
        log = get_logger()
        _emit(log.info, event="x", openweather_api_key="zzz")
        record = _last_json_line(capsys)
        assert record["openweather_api_key"] == "***"

    def test_redacts_nested_payload(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json=True, redact_keys=("password",))
        log = get_logger()
        _emit(
            log.info,
            event="x",
            credentials={"user": "alice", "password": "p4nd4"},
        )
        record = _last_json_line(capsys)
        assert record["credentials"]["password"] == "***"
        assert record["credentials"]["user"] == "alice"

    def test_case_insensitive(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json=True, redact_keys=("AUTHORIZATION",))
        log = get_logger()
        _emit(log.info, event="x", Authorization="Bearer xyz")
        record = _last_json_line(capsys)
        assert record["Authorization"] == "***"

    def test_redacts_inside_list_of_dicts(self, capsys: pytest.CaptureFixture[str]) -> None:
        """H2 regression: secrets inside `payload=[{...}, {...}]` must be masked."""
        configure_logging(level="INFO", json=True, redact_keys=("api_key",))
        log = get_logger()
        _emit(
            log.info,
            event="batch",
            payload=[
                {"id": 1, "api_key": "secret1"},
                {"id": 2, "api_key": "secret2"},
            ],
        )
        record = _last_json_line(capsys)
        assert record["payload"][0]["api_key"] == "***"
        assert record["payload"][1]["api_key"] == "***"
        assert record["payload"][0]["id"] == 1

    def test_redacts_deeply_nested(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Recursion must descend through dict-in-list-in-dict."""
        configure_logging(level="INFO", json=True, redact_keys=("password",))
        log = get_logger()
        _emit(
            log.info,
            event="x",
            outer={"users": [{"name": "alice", "password": "p1"}]},
        )
        record = _last_json_line(capsys)
        assert record["outer"]["users"][0]["password"] == "***"
        assert record["outer"]["users"][0]["name"] == "alice"


class TestContextvars:
    def test_bound_context_merged_into_event(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="INFO", json=True)
        clear_contextvars()
        bind_contextvars(run_id="abc-123", agent="data")
        try:
            log = get_logger()
            _emit(log.info, event="ingest_done")
            record = _last_json_line(capsys)
            assert record["run_id"] == "abc-123"
            assert record["agent"] == "data"
        finally:
            clear_contextvars()
