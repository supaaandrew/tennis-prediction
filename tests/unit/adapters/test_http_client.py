"""Unit tests for the concrete `HttpxClient` transport (§S7).

Exercise `get`/`post` against an httpx `MockTransport` — no real network — and
confirm the project's transport-neutral `HttpResponse` is returned.
"""

from __future__ import annotations

import httpx
import pytest

from tennis.adapters.http_client import HttpxClient
from tennis.core.contracts import HttpClient, HttpResponse


def _client(handler) -> HttpxClient:
    return HttpxClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestHttpxClient:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(_client(lambda req: httpx.Response(200)), HttpClient)

    def test_get_returns_http_response_with_body_and_headers(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert request.url.params.get("a") == "b"
            return httpx.Response(200, json={"ok": True}, headers={"x-test": "1"})

        resp = _client(handler).get("https://api.example.com/x", params={"a": "b"})
        assert isinstance(resp, HttpResponse)
        assert resp.status == 200
        assert resp.headers["x-test"] == "1"
        assert resp.json() == {"ok": True}

    def test_post_sends_json_body(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["content"] = request.content
            return httpx.Response(201, content=b"created")

        resp = _client(handler).post("https://api.example.com/y", json={"k": 1})
        assert seen["method"] == "POST"
        assert b'"k"' in seen["content"]  # type: ignore[operator]
        assert resp.status == 201
        assert resp.body == b"created"

    def test_non_2xx_is_returned_not_raised(self) -> None:
        # The transport returns responses verbatim; adapters decide what a 404
        # means — the client never raises for status.
        resp = _client(lambda req: httpx.Response(404)).get("https://x/y")
        assert resp.status == 404
