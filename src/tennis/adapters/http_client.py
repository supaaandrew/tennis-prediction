"""Concrete `HttpClient` transport — the production HTTP surface every source
adapter shares (§S7 composition root).

Adapters depend on the `core.contracts.HttpClient` Protocol, never on httpx
directly; this is the single place httpx is imported. All network egress happens
here. The underlying `httpx.Client` is injectable for tests (a
`httpx.MockTransport`-backed client exercises `get`/`post` with no real network).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from tennis.core.contracts import HttpResponse

if TYPE_CHECKING:
    from collections.abc import Mapping

_DEFAULT_TIMEOUT_S = 30.0


class HttpxClient:
    """httpx-backed `HttpClient` (core.contracts). Returns the project's
    transport-neutral `HttpResponse` so adapters never see an httpx type."""

    def __init__(
        self, *, default_timeout_s: float = _DEFAULT_TIMEOUT_S, client: httpx.Client | None = None
    ) -> None:
        self._timeout = default_timeout_s
        self._client = client if client is not None else httpx.Client()

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> HttpResponse:
        resp = self._client.get(
            url,
            params=dict(params) if params is not None else None,
            headers=dict(headers) if headers is not None else None,
            timeout=timeout_s if timeout_s is not None else self._timeout,
        )
        return _to_response(resp)

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> HttpResponse:
        resp = self._client.post(
            url,
            json=dict(json) if json is not None else None,
            headers=dict(headers) if headers is not None else None,
            timeout=timeout_s if timeout_s is not None else self._timeout,
        )
        return _to_response(resp)

    def close(self) -> None:
        """Release the underlying connection pool. Idempotent."""
        self._client.close()


def _to_response(resp: httpx.Response) -> HttpResponse:
    return HttpResponse(
        status=resp.status_code, headers=dict(resp.headers), body=resp.content
    )
