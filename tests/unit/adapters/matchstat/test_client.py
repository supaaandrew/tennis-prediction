"""Matchstat HTTP client tests — Codex finding 5.

Covers the load-bearing transport + state code: §T9 two-header auth,
§T6 monthly quota counter (persisted + stale-month reset + warn@400 +
raise@501), §T6 lookup cache (fresh-watermark hit + stale-watermark
refresh + bad-cursor fallthrough), and 401/403/429/5xx → typed adapter
errors. Drives the client via `httpx.MockTransport` so no real network
fires.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from tennis.adapters.http_client import HttpxClient
from tennis.adapters.matchstat.client import MatchstatClient
from tennis.core.clock import FrozenClock
from tennis.core.config import MatchstatSource
from tennis.core.errors import (
    MatchstatAuthError,
    MatchstatQuotaExhaustedError,
    MatchstatTierMismatchError,
    RateLimitError,
    UpstreamUnavailableError,
)
from tennis.storage.postgres.rows import IngestWatermarkRow


def _config(**overrides: Any) -> MatchstatSource:
    defaults: dict[str, Any] = {
        "enabled": True,
        "api_key_env": "MATCHSTAT_API_KEY",
        "host": "tennis-api-atp-wta-itf.p.rapidapi.com",
        "base_url": "https://tennis-api-atp-wta-itf.p.rapidapi.com",
        "monthly_quota": 5,
        "quota_warn_at": 4,
        "per_minute_throttle": 100,
        "rate_limit_rps": 1.0,
        "request_timeout_s": 30,
        "lookforward_days": 2,
        "tier_ids": (1, 2, 3, 4),
        "lookup_refresh_days": 7,
    }
    defaults.update(overrides)
    return MatchstatSource.model_validate(defaults)


class _FakeWatermarks:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], IngestWatermarkRow] = {}
        self.upserts: list[IngestWatermarkRow] = []
        self.get_raises = False

    def get(self, *, source: str, scope: str) -> IngestWatermarkRow | None:
        if self.get_raises:
            raise RuntimeError("simulated watermark read failure")
        return self.store.get((source, scope))

    def upsert(self, row: IngestWatermarkRow) -> IngestWatermarkRow:
        self.upserts.append(row)
        self.store[(row.source, row.scope)] = row
        return row


def _make_client(
    *,
    handler: Any,
    config: MatchstatSource | None = None,
    api_key: str | None = "test-key",
    clock: FrozenClock | None = None,
    watermarks: _FakeWatermarks | None = None,
) -> tuple[MatchstatClient, _FakeWatermarks]:
    transport = httpx.MockTransport(handler)
    http = HttpxClient(client=httpx.Client(transport=transport))
    wm = watermarks if watermarks is not None else _FakeWatermarks()
    clk = clock if clock is not None else FrozenClock(datetime(2026, 5, 30, tzinfo=UTC))
    client = MatchstatClient(
        config=config if config is not None else _config(),
        http=http,
        api_key=api_key,
        clock=clk,
        watermarks=wm,
        sleep=lambda _s: None,  # do not actually sleep in tests
        monotonic=lambda: 0.0,
    )
    return client, wm


# ---------------------------------------------------------------------------
# §T9 auth header injection
# ---------------------------------------------------------------------------
class TestT9AuthHeaders:
    def test_both_rapidapi_headers_on_every_request(self) -> None:
        seen_headers: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(dict(request.headers))
            return httpx.Response(200, json={"data": [], "hasNextPage": False})

        client, _ = _make_client(handler=handler)

        client.list_fixtures(target_date="2026-05-30")
        client.list_rankings_singles()
        client.search(query="alcaraz")

        assert len(seen_headers) == 3
        for h in seen_headers:
            assert h["x-rapidapi-key"] == "test-key"
            assert h["x-rapidapi-host"] == "tennis-api-atp-wta-itf.p.rapidapi.com"

    def test_missing_api_key_raises_lazily_not_at_construction(self) -> None:
        client, _ = _make_client(
            handler=lambda r: httpx.Response(200, json={"data": []}),
            api_key=None,
        )
        # Construction succeeded; only the call raises.
        with pytest.raises(MatchstatAuthError):
            client.list_fixtures(target_date="2026-05-30")


# ---------------------------------------------------------------------------
# §T9 / error mapping
# ---------------------------------------------------------------------------
class TestStatusErrorMapping:
    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures_raise_typed_error(self, status: int) -> None:
        client, _ = _make_client(
            handler=lambda r: httpx.Response(status, json={"message": "no"})
        )
        with pytest.raises(MatchstatAuthError):
            client.list_fixtures(target_date="2026-05-30")

    def test_429_after_retries_raises_rate_limit(self) -> None:
        client, _ = _make_client(
            handler=lambda r: httpx.Response(429, json={"message": "slow down"})
        )
        with pytest.raises(RateLimitError):
            client.list_fixtures(target_date="2026-05-30")

    def test_5xx_raises_upstream_unavailable(self) -> None:
        client, _ = _make_client(
            handler=lambda r: httpx.Response(500, json={"message": "boom"})
        )
        with pytest.raises(UpstreamUnavailableError):
            client.list_fixtures(target_date="2026-05-30")


# ---------------------------------------------------------------------------
# §T6 quota counter
# ---------------------------------------------------------------------------
class TestT6Quota:
    def _ok_handler(self) -> Any:
        return lambda r: httpx.Response(
            200, json={"data": [], "hasNextPage": False}
        )

    def test_counter_increments_and_persists(self) -> None:
        client, wm = _make_client(handler=self._ok_handler())
        client.list_fixtures(target_date="2026-05-30")
        client.list_fixtures(target_date="2026-05-31")
        row = wm.store.get(("matchstat", "quota"))
        assert row is not None
        assert row.cursor["count"] == 2
        assert row.cursor["month_key"] == "2026-05"

    def test_raises_before_501st_call(self) -> None:
        cfg = _config(monthly_quota=3, quota_warn_at=2)
        client, _ = _make_client(handler=self._ok_handler(), config=cfg)
        # 3 calls succeed; the 4th raises BEFORE issuing a request.
        for _ in range(3):
            client.list_fixtures(target_date="2026-05-30")
        with pytest.raises(MatchstatQuotaExhaustedError):
            client.list_fixtures(target_date="2026-05-30")

    def test_quota_loaded_from_persisted_watermark_on_fresh_process(self) -> None:
        # Pre-seed the watermark — a previous process counted 3 calls.
        wm = _FakeWatermarks()
        wm.store[("matchstat", "quota")] = IngestWatermarkRow(
            source="matchstat",
            scope="quota",
            last_processed_at=datetime(2026, 5, 30, tzinfo=UTC),
            cursor={"month_key": "2026-05", "count": 3},
        )
        cfg = _config(monthly_quota=4, quota_warn_at=3)
        client, _ = _make_client(
            handler=self._ok_handler(), config=cfg, watermarks=wm
        )
        # One more call lands at count=4; the next raises.
        client.list_fixtures(target_date="2026-05-30")
        with pytest.raises(MatchstatQuotaExhaustedError):
            client.list_fixtures(target_date="2026-05-30")

    def test_stale_month_resets_counter(self) -> None:
        wm = _FakeWatermarks()
        wm.store[("matchstat", "quota")] = IngestWatermarkRow(
            source="matchstat",
            scope="quota",
            last_processed_at=datetime(2026, 4, 30, tzinfo=UTC),
            cursor={"month_key": "2026-04", "count": 500},  # last month at cap
        )
        clock = FrozenClock(datetime(2026, 5, 1, 0, 5, tzinfo=UTC))
        client, _ = _make_client(
            handler=self._ok_handler(), watermarks=wm, clock=clock
        )
        # New UTC month → counter resets, so this call must succeed.
        client.list_fixtures(target_date="2026-05-01")
        row = wm.store[("matchstat", "quota")]
        assert row.cursor["month_key"] == "2026-05"
        assert row.cursor["count"] == 1

    def test_quota_remaining_property(self) -> None:
        cfg = _config(monthly_quota=10, quota_warn_at=8)
        client, _ = _make_client(handler=self._ok_handler(), config=cfg)
        assert client.quota_remaining == 10
        client.list_fixtures(target_date="2026-05-30")
        client.list_fixtures(target_date="2026-05-31")
        assert client.quota_remaining == 8


# ---------------------------------------------------------------------------
# §T6 lookup cache (H2 fix coverage — was untested at landing)
# ---------------------------------------------------------------------------
class TestT6LookupCache:
    def test_fresh_watermark_served_from_persisted_data_no_http(self) -> None:
        # Pre-seed a fresh lookup watermark with persisted data.
        wm = _FakeWatermarks()
        seeded_at = datetime(2026, 5, 30, 6, 0, tzinfo=UTC).isoformat()
        wm.store[("matchstat", "lookup:courts")] = IngestWatermarkRow(
            source="matchstat",
            scope="lookup:courts",
            last_processed_at=datetime(2026, 5, 30, 6, 0, tzinfo=UTC),
            cursor={"fetched_at": seeded_at, "data": [{"id": 1, "name": "Hard"}]},
        )

        call_count = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            return httpx.Response(200, json={"data": [{"id": 99, "name": "FromHttp"}]})

        # Clock is 2 days after seed — within 7-day refresh window.
        clock = FrozenClock(datetime(2026, 6, 1, 6, 0, tzinfo=UTC))
        client, _ = _make_client(handler=handler, watermarks=wm, clock=clock)

        courts = client.list_courts()
        assert call_count[0] == 0  # no HTTP call — served from cache
        assert len(courts) == 1
        assert courts[0].name == "Hard"

    def test_stale_watermark_triggers_refetch_and_repersists_data(self) -> None:
        wm = _FakeWatermarks()
        old_at = datetime(2026, 5, 1, 0, 0, tzinfo=UTC).isoformat()
        wm.store[("matchstat", "lookup:courts")] = IngestWatermarkRow(
            source="matchstat",
            scope="lookup:courts",
            last_processed_at=datetime(2026, 5, 1, tzinfo=UTC),
            cursor={"fetched_at": old_at, "data": [{"id": 1, "name": "Old"}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": 2, "name": "Fresh"}]})

        # Clock is 30 days after seed — past the 7-day refresh window.
        clock = FrozenClock(datetime(2026, 5, 31, tzinfo=UTC))
        client, _ = _make_client(handler=handler, watermarks=wm, clock=clock)

        courts = client.list_courts()
        assert len(courts) == 1
        assert courts[0].name == "Fresh"
        # Re-persisted: both fetched_at AND data
        row = wm.store[("matchstat", "lookup:courts")]
        assert "data" in row.cursor
        assert row.cursor["data"] == [{"id": 2, "name": "Fresh"}]

    def test_bad_cursor_schema_falls_through_to_fresh_fetch(self) -> None:
        wm = _FakeWatermarks()
        wm.store[("matchstat", "lookup:courts")] = IngestWatermarkRow(
            source="matchstat",
            scope="lookup:courts",
            last_processed_at=datetime(2026, 5, 30, tzinfo=UTC),
            cursor={"fetched_at": "not-a-real-iso", "data": [{"id": 1}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": 9, "name": "Fresh"}]})

        client, _ = _make_client(handler=handler, watermarks=wm)
        courts = client.list_courts()
        assert courts[0].name == "Fresh"

    def test_watermark_read_failure_falls_through_to_fresh_fetch(self) -> None:
        wm = _FakeWatermarks()
        wm.get_raises = True

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": 1, "name": "Hard"}]})

        client, _ = _make_client(handler=handler, watermarks=wm)
        courts = client.list_courts()
        assert len(courts) == 1
        assert courts[0].name == "Hard"

    def test_naive_fetched_at_in_cursor_treated_as_invalid(self) -> None:
        # tz-naive datetime in the cursor must NOT be trusted — fall through.
        wm = _FakeWatermarks()
        wm.store[("matchstat", "lookup:courts")] = IngestWatermarkRow(
            source="matchstat",
            scope="lookup:courts",
            last_processed_at=datetime(2026, 5, 30, tzinfo=UTC),
            cursor={
                "fetched_at": "2026-05-30T06:00:00",  # NO tzinfo
                "data": [{"id": 1, "name": "Stale"}],
            },
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": 2, "name": "Fresh"}]})

        client, _ = _make_client(handler=handler, watermarks=wm)
        courts = client.list_courts()
        assert courts[0].name == "Fresh"


# ---------------------------------------------------------------------------
# §T5 startup TourRank-id canary
# ---------------------------------------------------------------------------
class TestT5VerifyTierIds:
    """`verify_tier_ids` reads the §T6 lookup cache and fails loud on
    upstream renumbering. Warm-cache path costs 0 quota."""

    def _seed_ranks(
        self,
        wm: _FakeWatermarks,
        rows: list[dict[str, Any]],
        *,
        fetched_at: datetime = datetime(2026, 5, 30, 6, 0, tzinfo=UTC),
    ) -> None:
        wm.store[("matchstat", "lookup:ranks")] = IngestWatermarkRow(
            source="matchstat",
            scope="lookup:ranks",
            last_processed_at=fetched_at,
            cursor={"fetched_at": fetched_at.isoformat(), "data": rows},
        )

    def test_matching_tier_ids_pass_silently(self) -> None:
        """All four configured ids map to a known tier name → no raise,
        no extra HTTP (lookup cache is warm)."""
        wm = _FakeWatermarks()
        self._seed_ranks(wm, [
            {"id": 1, "name": "Grand Slam"},
            {"id": 2, "name": "Masters 1000"},
            {"id": 3, "name": "ATP 500"},
            {"id": 4, "name": "ATP 250"},
            {"id": 5, "name": "Challenger"},  # configured ids ignore extras
        ])
        call_count = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            return httpx.Response(500)  # would explode if hit

        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))  # within 7d
        client, _ = _make_client(handler=handler, watermarks=wm, clock=clock)
        client.verify_tier_ids()  # must not raise
        assert call_count[0] == 0  # warm-cache: zero HTTP

    def test_renumbered_tier_raises_typed_error(self) -> None:
        """If matchstat ever renumbers (id 2 returns a name we don't know),
        raise `MatchstatTierMismatchError` listing the offending id + name."""
        wm = _FakeWatermarks()
        self._seed_ranks(wm, [
            {"id": 1, "name": "Grand Slam"},
            {"id": 2, "name": "Renamed Tier"},  # divergence!
            {"id": 3, "name": "ATP 500"},
            {"id": 4, "name": "ATP 250"},
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)  # cache hit; should not fire

        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))
        client, _ = _make_client(handler=handler, watermarks=wm, clock=clock)
        with pytest.raises(MatchstatTierMismatchError) as exc_info:
            client.verify_tier_ids()
        msg = str(exc_info.value)
        assert "tier id 2" in msg
        assert "Renamed Tier" in msg

    def test_missing_tier_id_in_response_raises_typed_error(self) -> None:
        """A configured id that the live lookup doesn't expose at all is
        also a renumbering signal — fail loud (no silent acceptance)."""
        wm = _FakeWatermarks()
        # Tier 4 (ATP 250) is missing — `id_to_name.get(4)` returns None.
        self._seed_ranks(wm, [
            {"id": 1, "name": "Grand Slam"},
            {"id": 2, "name": "Masters 1000"},
            {"id": 3, "name": "ATP 500"},
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))
        client, _ = _make_client(handler=handler, watermarks=wm, clock=clock)
        with pytest.raises(MatchstatTierMismatchError) as exc_info:
            client.verify_tier_ids()
        assert "tier id 4" in str(exc_info.value)


# ---------------------------------------------------------------------------
# §T6 public monthly_quota accessor (Codex finding K leak fix)
# ---------------------------------------------------------------------------
class TestMonthlyQuotaAccessor:
    def test_monthly_quota_property_matches_config(self) -> None:
        cfg = _config(monthly_quota=500)
        client, _ = _make_client(
            handler=lambda r: httpx.Response(200, json={}),
            config=cfg,
        )
        assert client.monthly_quota == 500
