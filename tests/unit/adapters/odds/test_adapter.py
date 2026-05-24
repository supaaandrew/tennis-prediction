"""Unit tests for the Odds API adapter orchestration.

All repositories and the client are mocked; a FrozenClock pins "now". Tests
exercise control flow — name resolution, match linkage, p1/p2 perspective,
dead-letter-and-continue, failure-aware watermark, and the opening/closing
post-pass (§J1–§J3) — not HTTP or parsing (covered in test_client/test_parser).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tennis.adapters.odds.adapter import OddsApiAdapter
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import UpstreamUnavailableError
from tennis.core.ids import normalize_player_name, p1_player_id
from tennis.storage.postgres.rows import (
    IngestWatermarkRow,
    MatchRow,
    OddsSnapshotRow,
    PlayerAliasRow,
)

A_NAME = "Novak Djokovic"
B_NAME = "Rafael Nadal"
A_ID = 1001
B_ID = 2002
MATCH_ID = 500000  # even → p1_player_id picks the smaller id (A_ID) → p1_is_a
COMMENCE = "2024-06-01T13:00:00Z"
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

_NORM_A = normalize_player_name(A_NAME)
_NORM_B = normalize_player_name(B_NAME)


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    return load_config(config_path)


def _alias(player_id: int) -> PlayerAliasRow:
    return PlayerAliasRow(alias="n", source="odds_api", player_id=player_id, confidence="exact")


def _alias_lookup(*, alias: str, source: str) -> PlayerAliasRow | None:
    if alias == _NORM_A:
        return _alias(A_ID)
    if alias == _NORM_B:
        return _alias(B_ID)
    return None


def _match_row(*, match_id: int = MATCH_ID, start_ts: datetime | None = None) -> MatchRow:
    return MatchRow(
        match_id=match_id, tournament_id=7, round="QF", match_date=date(2024, 6, 1),
        p1_id=A_ID, p2_id=B_ID, status="scheduled", source="sackmann",
        source_uid="2024-x:1", start_ts=start_ts,
    )


def _event(**overrides) -> dict:
    base = {
        "id": "evt-1",
        "sport_key": "tennis_atp",
        "commence_time": COMMENCE,
        "home_team": A_NAME,
        "away_team": B_NAME,
        "bookmakers": [
            {
                "key": "pinnacle",
                "last_update": "2024-06-01T11:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": A_NAME, "price": 1.30},
                            {"name": B_NAME, "price": 3.80},
                        ],
                    }
                ],
            }
        ],
    }
    base.update(overrides)
    return base


def _snap(captured_at: datetime, method: str) -> OddsSnapshotRow:
    return OddsSnapshotRow(
        match_id=MATCH_ID, bookmaker="pinnacle", captured_at=captured_at,
        p1_decimal=1.30, p2_decimal=3.80, p1_implied=0.75, p2_implied=0.25,
        vig=0.03, devig_method=method, market="h2h", snapshot_id=99,
    )


def _build(config: AppConfig):
    client = MagicMock()
    matches = MagicMock()
    odds = MagicMock()
    aliases = MagicMock()
    watermarks = MagicMock()
    dead_letter = MagicMock()

    matches.find_by_players_and_date.return_value = _match_row()
    matches.get.return_value = _match_row()
    odds.list_for_match.return_value = []
    watermarks.get.return_value = None
    aliases.get.side_effect = _alias_lookup

    mocks = {
        "client": client, "matches": matches, "odds": odds, "aliases": aliases,
        "watermarks": watermarks, "dead_letter": dead_letter,
    }
    adapter = OddsApiAdapter(
        config=config, clock=FrozenClock(NOW), client=client, matches=matches,
        odds=odds, aliases=aliases, watermarks=watermarks, dead_letter=dead_letter,
    )
    return adapter, mocks


def _inserted(mocks) -> list[OddsSnapshotRow]:
    return [call.args[0] for call in mocks["odds"].insert.call_args_list]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_fetch_writes_two_rows_per_book(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["client"].fetch_current_odds.return_value = [_event()]

    result = adapter.fetch_upcoming()

    assert result.events_processed == 1
    assert result.events_skipped == 0
    assert result.snapshots_inserted == 2  # shin + proportional, one book
    assert result.matches_touched == 1
    assert result.failures == 0
    assert result.complete
    methods = {row.devig_method for row in _inserted(m)}
    assert methods == {"shin", "proportional"}


def test_perspective_uses_p1_player_id_not_outcome_order(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["client"].fetch_current_odds.return_value = [_event()]

    adapter.fetch_upcoming()

    # MATCH_ID even → p1 == A_ID → p1_decimal follows A's price (1.30).
    assert p1_player_id(MATCH_ID, A_ID, B_ID) == A_ID
    for row in _inserted(m):
        assert row.match_id == MATCH_ID
        assert row.p1_decimal == 1.30
        assert row.p2_decimal == 3.80
    # shin favorite implied prob exceeds the underdog's.
    shin = next(r for r in _inserted(m) if r.devig_method == "shin")
    assert shin.p1_implied > shin.p2_implied


def test_logit_is_never_emitted(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["client"].fetch_current_odds.return_value = [_event()]
    adapter.fetch_upcoming()
    methods = {row.devig_method for row in _inserted(m)}
    assert "logit" not in methods
    assert methods <= {"shin", "proportional"}


# ---------------------------------------------------------------------------
# Resolution / linkage data gaps — dead-lettered, NOT failures (J1, I2)
# ---------------------------------------------------------------------------
def test_unresolved_name_is_skipped_and_dead_lettered(config: AppConfig) -> None:
    adapter, m = _build(config)
    # B resolves to nothing.
    m["aliases"].get.side_effect = lambda *, alias, source: (
        _alias(A_ID) if alias == _NORM_A else None
    )
    m["client"].fetch_current_odds.return_value = [_event()]

    result = adapter.fetch_upcoming()

    assert result.events_skipped == 1
    assert result.snapshots_inserted == 0
    assert result.failures == 0  # data gap, not a failure
    assert result.complete  # watermark still completes
    m["odds"].insert.assert_not_called()
    dl = m["dead_letter"].append.call_args.args[0]
    assert dl.scope.startswith("player_resolution:")


def test_unlinked_match_is_skipped_and_dead_lettered(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["matches"].find_by_players_and_date.return_value = None
    m["client"].fetch_current_odds.return_value = [_event()]

    result = adapter.fetch_upcoming()

    assert result.events_skipped == 1
    assert result.failures == 0
    assert result.complete
    m["odds"].insert.assert_not_called()
    dl = m["dead_letter"].append.call_args.args[0]
    assert dl.scope.startswith("match_linkage:")


def test_never_fabricates_match_id_when_unlinked(config: AppConfig) -> None:
    # Regression for J1: no insert may happen without a real linked match.
    adapter, m = _build(config)
    m["matches"].find_by_players_and_date.return_value = None
    m["client"].fetch_current_odds.return_value = [_event()]
    adapter.fetch_upcoming()
    m["odds"].insert.assert_not_called()


# ---------------------------------------------------------------------------
# Per-event fault isolation — storage failure IS a failure (watermark blocked)
# ---------------------------------------------------------------------------
def test_storage_failure_isolated_and_counts_as_failure(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["client"].fetch_current_odds.return_value = [_event(id="evt-1"), _event(id="evt-2")]
    # First insert (event 1) raises; event 2's two inserts succeed.
    m["odds"].insert.side_effect = [RuntimeError("db down"), None, None]

    result = adapter.fetch_upcoming()

    assert result.events_processed == 2
    assert result.failures == 1
    assert not result.complete
    assert result.snapshots_inserted == 2  # event 2 still landed
    scopes = [c.args[0].scope for c in m["dead_letter"].append.call_args_list]
    assert any(s.startswith("insert:") for s in scopes)


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------
def test_fetch_watermark_scope_and_completion(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["client"].fetch_current_odds.return_value = [_event()]
    adapter.fetch_upcoming()
    wm = m["watermarks"].upsert.call_args.args[0]
    assert wm.source == "odds_api"
    assert wm.scope == "fetch:upcoming"
    assert wm.cursor["status"] == "complete"


def test_fetch_transport_failure_marks_incomplete(config: AppConfig) -> None:
    adapter, m = _build(config)
    m["client"].fetch_current_odds.side_effect = UpstreamUnavailableError("boom")

    result = adapter.fetch_upcoming()

    assert result.failures == 1
    assert not result.complete
    wm = m["watermarks"].upsert.call_args.args[0]
    assert wm.cursor["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Opening/closing post-pass (§J3)
# ---------------------------------------------------------------------------
def test_recompute_flags_sets_opening_and_closing(config: AppConfig) -> None:
    adapter, m = _build(config)
    start = datetime(2024, 6, 1, 13, 0, tzinfo=UTC)
    t_open = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)      # earliest
    t_close = datetime(2024, 6, 1, 12, 55, tzinfo=UTC)   # within 15 min of start
    rows = [_snap(t, method) for t in (t_open, t_close) for method in ("shin", "proportional")]
    m["matches"].get.return_value = _match_row(start_ts=start)
    m["odds"].list_for_match.return_value = rows

    adapter._recompute_flags(MATCH_ID)

    reinserted = _inserted(m)
    opens = [r for r in reinserted if r.is_opening]
    closes = [r for r in reinserted if r.is_closing]
    assert len(opens) == 2 and all(r.captured_at == t_open for r in opens)
    assert len(closes) == 2 and all(r.captured_at == t_close for r in closes)


def test_recompute_flags_no_closing_when_start_ts_none(config: AppConfig) -> None:
    adapter, m = _build(config)
    t1 = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
    t2 = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    m["matches"].get.return_value = _match_row(start_ts=None)
    m["odds"].list_for_match.return_value = [_snap(t1, "shin"), _snap(t2, "shin")]

    adapter._recompute_flags(MATCH_ID)

    reinserted = _inserted(m)
    assert all(not r.is_closing for r in reinserted)
    assert any(r.is_opening and r.captured_at == t1 for r in reinserted)


# ---------------------------------------------------------------------------
# Backfill — chain walk + failure-aware watermark
# ---------------------------------------------------------------------------
def test_backfill_walks_chain_and_scopes_by_year(config: AppConfig) -> None:
    adapter, m = _build(config)
    wrapper1 = {
        "timestamp": "2020-06-01T10:00:00Z",
        "next_timestamp": "2020-06-01T11:00:00Z",
        "data": [_event(commence_time="2020-06-01T13:00:00Z")],
    }
    wrapper2 = {"timestamp": "2020-06-01T11:00:00Z", "next_timestamp": None, "data": []}
    m["client"].fetch_historical_odds.side_effect = [wrapper1, wrapper2]

    result = adapter.backfill([2020])

    assert result.years_processed == 1
    assert result.snapshots_inserted == 2
    assert m["client"].fetch_historical_odds.call_count == 2  # followed next_timestamp
    wm = m["watermarks"].upsert.call_args.args[0]
    assert wm.scope == "backfill:2020"
    assert wm.cursor["status"] == "complete"
    assert wm.cursor["last_timestamp"] == "2020-06-01T11:00:00Z"


def test_backfill_from_config_covers_coverage_start_to_current_year(config: AppConfig) -> None:
    adapter, m = _build(config)
    # Empty wrapper with no next_timestamp → exactly one fetch per year.
    m["client"].fetch_historical_odds.return_value = {
        "timestamp": "2020-01-01T00:00:00Z", "next_timestamp": None, "data": [],
    }

    result = adapter.backfill_from_config()

    # config coverage_start_year=2020 .. NOW.year=2024 inclusive → 5 years.
    assert config.sources.odds_api.coverage_start_year == 2020
    assert result.years_processed == 5
    scopes = {c.args[0].scope for c in m["watermarks"].upsert.call_args_list}
    assert scopes == {f"backfill:{year}" for year in range(2020, 2025)}


def test_backfill_transport_failure_preserves_prior_watermark(config: AppConfig) -> None:
    adapter, m = _build(config)
    prior = IngestWatermarkRow(
        source="odds_api", scope="backfill:2020",
        last_processed_at=datetime(2024, 1, 1, tzinfo=UTC),
        cursor={"last_timestamp": "2020-03-01T00:00:00Z", "status": "complete"},
    )
    m["watermarks"].get.return_value = prior
    m["client"].fetch_historical_odds.side_effect = UpstreamUnavailableError("boom")

    result = adapter.backfill([2020])

    assert result.failures == 1
    assert not result.complete
    wm = m["watermarks"].upsert.call_args.args[0]
    assert wm.cursor["status"] == "incomplete"
    # Never advanced over the hole — prior resume point preserved.
    assert wm.cursor["last_timestamp"] == "2020-03-01T00:00:00Z"


def test_backfill_non_advancing_cursor_terminates(config: AppConfig) -> None:
    # CRITICAL guard: an upstream chain whose next_timestamp repeats would
    # loop _walk_year forever. return_value (not side_effect) repeats the same
    # wrapper indefinitely, so if the guard regressed this test would hang.
    adapter, m = _build(config)
    m["client"].fetch_historical_odds.return_value = {
        "timestamp": "2020-06-01T10:00:00Z",
        "next_timestamp": "2020-06-01T10:00:00Z",  # never advances
        "data": [],
    }

    result = adapter.backfill([2020])

    assert result.failures == 1
    assert not result.complete
    dl = m["dead_letter"].append.call_args_list
    assert len(dl) == 1
    assert dl[0].args[0].error["type"] == "NonAdvancingCursor"


def test_backfill_malformed_wrapper_is_counted_failure(config: AppConfig) -> None:
    # HIGH guard: a non-dict historical payload must become a counted failure
    # + dead-letter, not an uncaught AttributeError that aborts the run.
    adapter, m = _build(config)
    m["client"].fetch_historical_odds.return_value = [{"not": "a wrapper"}]

    result = adapter.backfill([2020])

    assert result.failures == 1
    assert not result.complete
    dl = m["dead_letter"].append.call_args_list
    assert len(dl) == 1
    assert dl[0].args[0].error["type"] == "MalformedHistoricalWrapper"
