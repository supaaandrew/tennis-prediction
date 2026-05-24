"""Unit tests for the OWM adapter orchestration.

All repositories and the OWM client are mocked; a FrozenClock pins the date
range so backfill spans only a handful of days. Tests exercise control flow
(venue skipping, watermark gating + advance, dead-letter-and-continue,
forecast horizon gating) rather than HTTP or parsing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from tennis.adapters.owm.adapter import OWMAdapter
from tennis.adapters.owm.client import HttpOWMClient
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import HttpResponse
from tennis.storage.postgres.rows import IngestWatermarkRow, VenueRow

# Config season start is 2000; pin "today" a few days in so the range is tiny.
BACKFILL_NOW = datetime(2000, 1, 3, 6, 30, tzinfo=UTC)
FORECAST_NOW = datetime(2026, 5, 23, 6, 0, tzinfo=UTC)

VENUE = VenueRow(
    venue_id=1, city="Melbourne", country_code="AUS",
    latitude=-37.81, longitude=144.96,
)


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    return load_config(config_path)


def _build(config: AppConfig, *, clock: FrozenClock | None = None):
    client = MagicMock()
    client.fetch_day_summary.return_value = {"temperature": {"afternoon": 300.0}}
    client.fetch_forecast.return_value = {"hourly": []}

    mocks = {
        "client": client,
        "venues": MagicMock(),
        "weather": MagicMock(),
        "watermarks": MagicMock(),
        "dead_letter": MagicMock(),
    }
    mocks["venues"].get.return_value = VENUE
    mocks["watermarks"].get.return_value = None

    adapter = OWMAdapter(
        config=config,
        clock=clock or FrozenClock(BACKFILL_NOW),
        client=client,
        venues=mocks["venues"],
        weather=mocks["weather"],
        watermarks=mocks["watermarks"],
        dead_letter=mocks["dead_letter"],
    )
    return adapter, mocks


def _backfill_days(client: MagicMock) -> list[date]:
    return [call.args[2] for call in client.fetch_day_summary.call_args_list]


# ---------------------------------------------------------------------------
# Venue coordinate gating
# ---------------------------------------------------------------------------
def test_venue_missing_coords_is_skipped_not_dead_lettered(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["venues"].get.return_value = VenueRow(
        venue_id=1, city="Nowhere", country_code="XXX",
        latitude=None, longitude=None,
    )

    result = adapter.backfill([1])

    assert result.venues_skipped == 1
    assert result.venues_processed == 0
    mocks["dead_letter"].append.assert_not_called()
    mocks["client"].fetch_day_summary.assert_not_called()


def test_unknown_venue_is_skipped(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["venues"].get.return_value = None

    result = adapter.backfill([99])

    assert result.venues_skipped == 1
    mocks["dead_letter"].append.assert_not_called()


# ---------------------------------------------------------------------------
# Watermark gating + advance
# ---------------------------------------------------------------------------
def test_watermark_gates_backfill(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["watermarks"].get.return_value = IngestWatermarkRow(
        source="owm", scope="backfill:1", last_processed_at=BACKFILL_NOW,
        cursor={"status": "complete", "last_date": "2000-01-01"},
    )

    adapter.backfill([1])

    # 2000-01-01 already covered; only 01-02 and 01-03 are fetched.
    assert _backfill_days(mocks["client"]) == [date(2000, 1, 2), date(2000, 1, 3)]


def test_backfill_skipped_when_already_current(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["watermarks"].get.return_value = IngestWatermarkRow(
        source="owm", scope="backfill:1", last_processed_at=BACKFILL_NOW,
        cursor={"status": "complete", "last_date": "2000-01-03"},
    )

    adapter.backfill([1])

    mocks["client"].fetch_day_summary.assert_not_called()


def test_watermark_advanced_after_successful_backfill(config: AppConfig) -> None:
    adapter, mocks = _build(config)

    result = adapter.backfill([1])

    assert result.complete is True
    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.cursor["status"] == "complete"
    assert row.cursor["last_date"] == "2000-01-03"


# ---------------------------------------------------------------------------
# Dead-letter + continue, failure-aware watermark
# ---------------------------------------------------------------------------
def test_dead_letter_on_failure_and_backfill_continues(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    # Range is 2000-01-01..01-03 (3 days): fail the middle day, succeed others.
    mocks["client"].fetch_day_summary.side_effect = [
        {"temperature": {"afternoon": 300.0}},
        RuntimeError("upstream boom"),
        {"temperature": {"afternoon": 301.0}},
    ]

    result = adapter.backfill([1])

    assert result.failures == 1
    assert result.observations_upserted == 2  # the two good days still stored
    assert mocks["weather"].upsert.call_count == 2
    mocks["dead_letter"].append.assert_called_once()
    # Watermark resume point is the last contiguous success — day 1 — so the
    # hole at day 2 (and the post-hole day 3) are retried next run.
    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.cursor["status"] == "incomplete"
    assert row.cursor["last_date"] == "2000-01-01"


def test_watermark_not_advanced_on_failure(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["client"].fetch_day_summary.side_effect = RuntimeError("boom")

    result = adapter.backfill([1])

    assert result.complete is False
    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.cursor["status"] == "incomplete"
    # No prior watermark and zero successes this run: omit ``last_date``
    # entirely so the next run resumes from the config season start.
    assert "last_date" not in row.cursor


def test_watermark_preserves_prior_last_date_on_total_failure(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    # Prior run successfully covered up to 2000-01-02; today is 2000-01-03.
    mocks["watermarks"].get.return_value = IngestWatermarkRow(
        source="owm", scope="backfill:1", last_processed_at=BACKFILL_NOW,
        cursor={"status": "complete", "last_date": "2000-01-02"},
    )
    mocks["client"].fetch_day_summary.side_effect = RuntimeError("boom")

    adapter.backfill([1])

    # The only new date (2000-01-03) failed; prior progress must survive.
    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.cursor["status"] == "incomplete"
    assert row.cursor["last_date"] == "2000-01-02"


# ---------------------------------------------------------------------------
# Forecast horizon gating
# ---------------------------------------------------------------------------
def test_forecast_beyond_max_horizon_is_skipped(config: AppConfig) -> None:
    adapter, mocks = _build(config, clock=FrozenClock(FORECAST_NOW))
    base = int(FORECAST_NOW.timestamp())

    def _hour(horizon_h: int) -> dict[str, object]:
        return {
            "dt": base + horizon_h * 3600, "temp": 300.0, "humidity": 50,
            "wind_speed": 3.0, "wind_deg": 90, "pressure": 1010, "clouds": 10,
        }

    mocks["client"].fetch_forecast.return_value = {
        "hourly": [_hour(1), _hour(200)]  # 1h kept (low), 200h beyond max
    }

    result = adapter.fetch_forecasts([1])

    assert result.observations_upserted == 1
    assert result.horizon_skipped == 1
    assert mocks["weather"].upsert.call_count == 1


def test_forecast_dead_letters_on_client_failure(config: AppConfig) -> None:
    adapter, mocks = _build(config, clock=FrozenClock(FORECAST_NOW))
    mocks["client"].fetch_forecast.side_effect = RuntimeError("boom")

    result = adapter.fetch_forecasts([1])

    assert result.failures == 1
    mocks["dead_letter"].append.assert_called_once()


# ---------------------------------------------------------------------------
# Repository fault isolation (Codex FIX 1)
# ---------------------------------------------------------------------------
def test_venue_repo_exception_dead_letters_and_continues(config: AppConfig) -> None:
    """A VenueRepository.get() exception must NOT abort the run — the venue
    is dead-lettered, logged, and processing continues with the next venue."""
    adapter, mocks = _build(config)
    mocks["venues"].get.side_effect = RuntimeError("db connection reset")

    result = adapter.backfill([1, 2])

    # Both venues skipped (no coords) but neither aborted the run.
    assert result.venues_skipped == 2
    assert result.venues_processed == 0
    # Both routed to dead_letter with venue_id + exception metadata.
    assert mocks["dead_letter"].append.call_count == 2
    dl_rows = [call.args[0] for call in mocks["dead_letter"].append.call_args_list]
    assert {row.payload["venue_id"] for row in dl_rows} == {1, 2}
    assert all(row.scope.startswith("venue_resolution:") for row in dl_rows)
    assert all(row.error["type"] == "RuntimeError" for row in dl_rows)
    # No fetch ever happened — we never got past coord resolution.
    mocks["client"].fetch_day_summary.assert_not_called()


def test_venue_repo_exception_also_isolated_in_forecast(config: AppConfig) -> None:
    adapter, mocks = _build(config, clock=FrozenClock(FORECAST_NOW))
    mocks["venues"].get.side_effect = RuntimeError("db down")

    result = adapter.fetch_forecasts([1, 2, 3])

    assert result.venues_skipped == 3
    assert mocks["dead_letter"].append.call_count == 3
    mocks["client"].fetch_forecast.assert_not_called()


# ---------------------------------------------------------------------------
# Forecast partial-venue-ingestion observability (Codex FIX 2)
# ---------------------------------------------------------------------------
def test_forecast_partial_venue_ingestion_warns_and_records_count(
    config: AppConfig,
) -> None:
    """v1 has no per-batch transaction; a mid-loop upsert failure leaves
    earlier hours committed. The adapter must (a) log a warning naming the
    partial-write count and (b) include that count in the dead-letter payload
    so an operator can decide whether to reconcile."""
    adapter, mocks = _build(config, clock=FrozenClock(FORECAST_NOW))
    base = int(FORECAST_NOW.timestamp())

    def _hour(h: int) -> dict[str, object]:
        return {
            "dt": base + h * 3600, "temp": 300.0, "humidity": 50,
            "wind_speed": 3.0, "wind_deg": 90, "pressure": 1010, "clouds": 10,
        }

    mocks["client"].fetch_forecast.return_value = {
        "hourly": [_hour(1), _hour(2), _hour(3)]
    }
    # Hour 3 fails to upsert; hours 1 and 2 are already persisted.
    mocks["weather"].upsert.side_effect = [None, None, RuntimeError("db write fail")]

    with capture_logs() as logs:
        result = adapter.fetch_forecasts([1])

    assert result.failures == 1
    assert mocks["weather"].upsert.call_count == 3  # tried all three
    # Warning fired with the partial count.
    partial_warnings = [
        e for e in logs if e.get("event") == "owm_forecast_partial_venue_ingestion"
    ]
    assert len(partial_warnings) == 1
    assert partial_warnings[0]["rows_written"] == 2
    # Dead-letter payload records the partial count so it's recoverable.
    dl_row = mocks["dead_letter"].append.call_args.args[0]
    assert dl_row.payload["rows_written_before_failure"] == 2


def test_forecast_failure_before_any_upsert_does_not_warn(config: AppConfig) -> None:
    """Pre-upsert failures (client raised, parse raised before any write)
    must NOT emit the partial-ingestion warning — there's no partial state."""
    adapter, mocks = _build(config, clock=FrozenClock(FORECAST_NOW))
    mocks["client"].fetch_forecast.side_effect = RuntimeError("network")

    with capture_logs() as logs:
        adapter.fetch_forecasts([1])

    assert not any(
        e.get("event") == "owm_forecast_partial_venue_ingestion" for e in logs
    )
    dl_row = mocks["dead_letter"].append.call_args.args[0]
    assert dl_row.payload["rows_written_before_failure"] == 0


# ---------------------------------------------------------------------------
# Dead-letter payload must never carry the API key (Codex FIX 4 / end-to-end)
# ---------------------------------------------------------------------------
def test_dead_letter_payload_does_not_leak_api_key(config: AppConfig) -> None:
    """Use a real HttpOWMClient (not a mock) so we exercise the actual error
    paths that touch ``appid`` query params. The 503 response forces an
    UpstreamUnavailableError per backfill day; the adapter wraps it into a
    DeadLetterRow. The serialized row — payload, error dict, source, scope —
    must never contain the API key."""
    leak_canary = "leak-canary-9f8e7d-do-not-log"
    fake_http = MagicMock()
    fake_http.get.return_value = HttpResponse(status=503, headers={}, body=b"{}")
    real_client = HttpOWMClient(
        config=config, http=fake_http, api_key=leak_canary,
        sleep=lambda _s: None, monotonic=lambda: 0.0,
    )

    venues = MagicMock(); venues.get.return_value = VENUE
    weather = MagicMock()
    watermarks = MagicMock(); watermarks.get.return_value = None
    dead_letter = MagicMock()

    adapter = OWMAdapter(
        config=config, clock=FrozenClock(BACKFILL_NOW),
        client=real_client, venues=venues, weather=weather,
        watermarks=watermarks, dead_letter=dead_letter,
    )
    adapter.backfill([1])

    dead_letter.append.assert_called()
    for call in dead_letter.append.call_args_list:
        row = call.args[0]
        serialized = repr((row.payload, row.error, row.source, row.scope))
        assert leak_canary not in serialized, (
            "API key leaked into a DeadLetterRow — check client error messages."
        )
