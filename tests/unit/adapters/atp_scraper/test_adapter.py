"""Unit tests for the ATP scraper adapter orchestration.

All repositories are mocked; the real client is replaced with a stub returning
fixture HTML (so the pure parser runs, but no HTTP/DB). A FrozenClock pins
"now". Tests exercise control flow — §K1 reconciliation, §K2 source_uid, §K3
match_date hashing, player resolution, the within-source intraday audit, the
fault split, and the failure-aware watermark.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tennis.adapters.atp_scraper.adapter import AtpScraperAdapter
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.ids import (
    match_id as compute_match_id,
    player_id_from_source,
    tournament_id as compute_tournament_id,
)
from tennis.storage.postgres.rows import MatchRow, PlayerAliasRow, PlayerRow

# Chosen so only Wimbledon (starts 2026-06-29) is inside the 21-day lookback
# window; Eastbourne (2026-06-22) falls before the cutoff and is filtered out
# before its (empty in tests) page is fetched. Keeps default builds to one
# tournament / two matches and avoids the FIX-1 zero-parse guard firing on the
# stub empty page.
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
WIM_SLUG = "wimbledon"
WIM_START = date(2026, 6, 29)
WIM_TID = compute_tournament_id(season=2026, slug=WIM_SLUG)

DJOKOVIC_SLUG = "novak-djokovic"
ALCARAZ_SLUG = "carlos-alcaraz"


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    return load_config(config_path)


def _shadow_id(slug: str) -> int:
    return player_id_from_source(source="atp_scraper", source_uid=slug)


def _wim_only(matches_html: str) -> Callable[[str], str]:
    """A fetch_tournament_matches stub: real HTML for Wimbledon, empty else."""

    def _fetch(slug: str) -> str:
        return matches_html if slug == WIM_SLUG else "<html></html>"

    return _fetch


def _build(config: AppConfig, load_fixture: Callable[[str], str],
           *, matches_fixture: str = "tournament_matches.html"):
    client = MagicMock()
    client.fetch_tournament_index.return_value = load_fixture("index.html")
    client.fetch_tournament_matches.side_effect = _wim_only(load_fixture(matches_fixture))

    players = MagicMock()
    aliases = MagicMock()
    tournaments = MagicMock()
    matches = MagicMock()
    watermarks = MagicMock()
    dead_letter = MagicMock()

    # Defaults: everything absent → shadow + stub + upsert paths.
    players.get_by_source.return_value = None
    aliases.get.return_value = None
    tournaments.get_by_season_slug.return_value = None
    matches.get.return_value = None

    mocks = {
        "client": client, "players": players, "aliases": aliases,
        "tournaments": tournaments, "matches": matches,
        "watermarks": watermarks, "dead_letter": dead_letter,
    }
    adapter = AtpScraperAdapter(
        config=config, clock=FrozenClock(NOW), client=client, players=players,
        aliases=aliases, tournaments=tournaments, matches=matches,
        watermarks=watermarks, dead_letter=dead_letter,
    )
    return adapter, mocks


def _upserted(mocks) -> list[MatchRow]:
    return [c.args[0] for c in mocks["matches"].upsert.call_args_list]


def _watermark_cursor(mocks):
    return mocks["watermarks"].upsert.call_args.args[0].cursor


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestFetchHappyPath:
    def test_processes_and_writes_wimbledon_matches(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        result = adapter.fetch()
        # Two QF matches written (the unassigned SF row is dropped by the parser).
        assert result.matches_written == 2
        assert result.failures == 0
        assert result.complete
        assert len(_upserted(mocks)) == 2

    def test_watermark_complete_on_clean_run(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        adapter.fetch()
        assert _watermark_cursor(mocks)["status"] == "complete"


# ---------------------------------------------------------------------------
# §K3 — match_date hashing convention + cross-source match_id agreement
# ---------------------------------------------------------------------------
class TestMatchDateConvention:
    def test_match_date_is_tournament_week_start(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        adapter.fetch()
        for row in _upserted(mocks):
            assert row.match_date == WIM_START  # NOT the real match day

    def test_match_id_agrees_with_sackmann_hash(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # Sackmann already knows both players (aliases resolve to canonical ids).
        sack = {"novak djokovic": 5001, "carlos alcaraz": 6002}

        def _alias(*, alias: str, source: str):
            if source == "sackmann" and alias in sack:
                return PlayerAliasRow(
                    alias=alias, source="sackmann",
                    player_id=sack[alias], confidence="exact",
                )
            return None

        adapter, mocks = _build(config, load_fixture)
        mocks["aliases"].get.side_effect = _alias
        adapter.fetch()

        # The scraper's match_id must equal the id a Sackmann ingest computes
        # for the same logical match (same tid/round/players/week-start date).
        sackmann_mid = compute_match_id(
            tournament_id=WIM_TID, round="QF",
            player_a=5001, player_b=6002, match_date=WIM_START,
        )
        written_ids = {r.match_id for r in _upserted(mocks)}
        assert sackmann_mid in written_ids


# ---------------------------------------------------------------------------
# §K2 — source_uid format
# ---------------------------------------------------------------------------
class TestSourceUid:
    def test_source_uid_format_and_slug_sorting(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        adapter.fetch()
        uids = {r.source_uid for r in _upserted(mocks)}
        # Slugs sorted: alcaraz < djokovic, regardless of page order.
        assert f"wimbledon:2026:QF:{ALCARAZ_SLUG}:{DJOKOVIC_SLUG}" in uids
        for row in _upserted(mocks):
            assert row.source == "atp_scraper"
            assert row.match_date_source == "atp_scraper"


# ---------------------------------------------------------------------------
# §K1 — reconciliation on the shared match_id
# ---------------------------------------------------------------------------
class TestReconciliation:
    def test_absent_match_is_full_upsert(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)  # matches.get → None
        adapter.fetch()
        assert mocks["matches"].upsert.call_count == 2
        mocks["matches"].update_live_fields.assert_not_called()

    def test_existing_final_is_skipped(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["matches"].get.return_value = MatchRow(
            match_id=1, tournament_id=WIM_TID, round="QF", match_date=WIM_START,
            p1_id=1, p2_id=2, status="final", source="sackmann", source_uid="x:1",
        )
        result = adapter.fetch()
        # Sackmann's authoritative final row is never touched.
        mocks["matches"].upsert.assert_not_called()
        mocks["matches"].update_live_fields.assert_not_called()
        assert result.matches_written == 0

    def test_existing_nonfinal_uses_update_live_fields(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["matches"].get.return_value = MatchRow(
            match_id=1, tournament_id=WIM_TID, round="QF", match_date=WIM_START,
            p1_id=1, p2_id=2, status="scheduled", source="atp_scraper",
            source_uid="wimbledon:2026:QF:a:b",
        )
        adapter.fetch()
        mocks["matches"].upsert.assert_not_called()
        assert mocks["matches"].update_live_fields.call_count == 2
        kwargs = mocks["matches"].update_live_fields.call_args.kwargs
        assert kwargs["match_date_source"] == "atp_scraper"
        assert kwargs["status"] in {"scheduled", "final"}


# ---------------------------------------------------------------------------
# Player resolution (§K: slug → Sackmann alias → shadow)
# ---------------------------------------------------------------------------
class TestPlayerResolution:
    def test_creates_shadow_when_unknown(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        adapter.fetch()
        # Four distinct players across two matches → four shadow upserts.
        shadow_rows = [c.args[0] for c in mocks["players"].upsert.call_args_list]
        assert all(isinstance(r, PlayerRow) and r.source == "atp_scraper" for r in shadow_rows)
        slugs = {r.source_uid for r in shadow_rows}
        assert DJOKOVIC_SLUG in slugs and ALCARAZ_SLUG in slugs
        # Each shadow keeps the deterministic hashed id.
        dj = next(r for r in shadow_rows if r.source_uid == DJOKOVIC_SLUG)
        assert dj.player_id == _shadow_id(DJOKOVIC_SLUG)

    def test_reuses_existing_shadow_without_recreating(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["players"].get_by_source.return_value = PlayerRow(
            player_id=999, full_name="X", source="atp_scraper", source_uid="any",
        )
        adapter.fetch()
        mocks["players"].upsert.assert_not_called()  # already exists

    def test_reuses_sackmann_player_id(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        sack = {"novak djokovic": 5001, "carlos alcaraz": 6002,
                "jannik sinner": 7003, "alexander zverev": 8004}

        def _alias(*, alias: str, source: str):
            if source == "sackmann" and alias in sack:
                return PlayerAliasRow(alias=alias, source="sackmann",
                                      player_id=sack[alias], confidence="exact")
            return None

        adapter, mocks = _build(config, load_fixture)
        mocks["aliases"].get.side_effect = _alias
        adapter.fetch()
        # No shadow players minted — Sackmann ids reused.
        mocks["players"].upsert.assert_not_called()
        written_player_ids = set()
        for r in _upserted(mocks):
            written_player_ids |= {r.p1_id, r.p2_id}
        assert {5001, 6002} <= written_player_ids

    def test_alias_collision_does_not_overwrite(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # An existing atp_scraper alias for a name pointing at a DIFFERENT
        # player must NOT be overwritten (I3 guard).
        def _alias(*, alias: str, source: str):
            if source == "atp_scraper":
                return PlayerAliasRow(alias=alias, source="atp_scraper",
                                      player_id=123456, confidence="exact")
            return None

        adapter, mocks = _build(config, load_fixture)
        mocks["aliases"].get.side_effect = _alias
        adapter.fetch()
        mocks["aliases"].upsert.assert_not_called()  # never clobbered


# ---------------------------------------------------------------------------
# Tournament stub creation
# ---------------------------------------------------------------------------
class TestTournamentStub:
    def test_creates_stub_when_absent(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        adapter.fetch()
        stubs = [c.args[0] for c in mocks["tournaments"].upsert.call_args_list]
        wim = next(t for t in stubs if t.slug == WIM_SLUG)
        assert wim.tier == "GS"          # draw_size 128 → GS
        assert wim.surface == "Grass"
        assert wim.venue_id is None
        assert wim.start_date == WIM_START

    def test_stub_surface_defaults_to_hard_when_unknown(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # Eastbourne in the index has no surface → stub defaults to Hard.
        # Widen the lookback (earlier "now") so Eastbourne is in-window, and
        # give it a non-empty page so the FIX-1 zero-parse guard doesn't fire.
        adapter, mocks = _build(config, load_fixture)
        adapter._clock = FrozenClock(datetime(2026, 6, 30, 12, 0, tzinfo=UTC))
        mocks["client"].fetch_tournament_matches.side_effect = None
        mocks["client"].fetch_tournament_matches.return_value = load_fixture(
            "tournament_matches.html"
        )
        adapter.fetch()
        stubs = [c.args[0] for c in mocks["tournaments"].upsert.call_args_list]
        eastbourne = next(t for t in stubs if t.slug == "eastbourne")
        assert eastbourne.surface == "Hard"

    def test_reuses_existing_tournament(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        existing = MagicMock()
        existing.tournament_id = WIM_TID
        mocks["tournaments"].get_by_season_slug.return_value = existing
        adapter.fetch()
        mocks["tournaments"].upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Lookback window filter
# ---------------------------------------------------------------------------
class TestLookbackFilter:
    def test_skips_tournaments_before_cutoff(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        # Move "now" far past both tournaments so both fall before the cutoff.
        adapter._clock = FrozenClock(datetime(2026, 9, 1, 12, 0, tzinfo=UTC))
        result = adapter.fetch()
        assert result.tournaments_processed == 0
        mocks["client"].fetch_tournament_matches.assert_not_called()


# ---------------------------------------------------------------------------
# Within-source intraday conflict (A10 / I4)
# ---------------------------------------------------------------------------
class TestIntradayConflict:
    def test_flags_shared_player_same_date(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(
            config, load_fixture, matches_fixture="intraday_conflict.html"
        )
        result = adapter.fetch()
        # Djokovic appears in both matches on the same tournament-week date.
        assert result.conflicts_flagged == 2
        assert mocks["matches"].mark_intraday_conflict.call_count == 2

    def test_no_conflict_for_distinct_players(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)  # default fixture, no overlap
        result = adapter.fetch()
        assert result.conflicts_flagged == 0
        mocks["matches"].mark_intraday_conflict.assert_not_called()


# ---------------------------------------------------------------------------
# Fault split (I2) — skips vs failures, failure-aware watermark
# ---------------------------------------------------------------------------
class TestFaultSplit:
    def test_index_fetch_failure_blocks_watermark(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["client"].fetch_tournament_index.side_effect = RuntimeError("boom")
        result = adapter.fetch()
        assert result.failures == 1
        assert result.tournaments_processed == 0
        assert _watermark_cursor(mocks)["status"] == "incomplete"

    def test_page_fetch_failure_is_failure(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["client"].fetch_tournament_matches.side_effect = RuntimeError("503")
        result = adapter.fetch()
        assert result.failures >= 1
        assert not result.complete
        assert mocks["dead_letter"].append.called

    def test_naive_ts_row_skipped_valid_row_still_processed(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # Per-row isolation (Codex HIGH): one naive-ts row is skipped +
        # dead-lettered, but the other valid row on the SAME page is processed.
        adapter, mocks = _build(
            config, load_fixture, matches_fixture="naive_start_ts.html"
        )
        result = adapter.fetch()
        assert result.matches_written == 1   # valid row processed
        assert result.matches_skipped == 1   # naive row skipped, page not aborted
        assert result.failures == 0
        assert result.complete
        assert _watermark_cursor(mocks)["status"] == "complete"
        assert mocks["dead_letter"].append.called

    def test_storage_failure_on_write_blocks_watermark(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["matches"].upsert.side_effect = RuntimeError("db down")
        result = adapter.fetch()
        assert result.failures >= 1
        assert _watermark_cursor(mocks)["status"] == "incomplete"

    def test_resolution_storage_failure_blocks_watermark(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        adapter, mocks = _build(config, load_fixture)
        mocks["players"].get_by_source.side_effect = RuntimeError("db down")
        result = adapter.fetch()
        assert result.failures >= 1
        assert not result.complete

    def test_player_resolution_error_is_skip_not_failure(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # A PlayerResolutionError during resolve is a data gap (I2): skipped,
        # dead-lettered, and it must NOT block watermark completion.
        from tennis.core.errors import PlayerResolutionError

        adapter, mocks = _build(config, load_fixture)
        mocks["players"].get_by_source.side_effect = PlayerResolutionError("ambiguous")
        result = adapter.fetch()
        assert result.failures == 0
        assert result.matches_skipped >= 1
        assert result.complete
        assert _watermark_cursor(mocks)["status"] == "complete"
        assert mocks["dead_letter"].append.called


# ---------------------------------------------------------------------------
# Zero-parse guard (Codex CRITICAL) + audit-failure decoupling (Codex MEDIUM)
# ---------------------------------------------------------------------------
class TestZeroParseGuard:
    def test_zero_parsed_matches_is_failure_not_silent_success(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # HTML drift → selectors match nothing → empty parse. Must NOT complete
        # cleanly with zero matches (that would silently mask a dropped feed).
        adapter, mocks = _build(config, load_fixture)
        mocks["client"].fetch_tournament_matches.side_effect = None
        mocks["client"].fetch_tournament_matches.return_value = (
            "<html><body>totally different markup</body></html>"
        )
        result = adapter.fetch()
        assert result.matches_written == 0
        assert result.failures >= 1
        assert not result.complete
        assert _watermark_cursor(mocks)["status"] == "incomplete"
        appended = [c.args[0] for c in mocks["dead_letter"].append.call_args_list]
        assert any(dl.error.get("type") == "ZeroParsedMatches" for dl in appended)


class TestAuditFailureDecoupling:
    def test_intraday_flag_failure_does_not_block_watermark(
        self, config: AppConfig, load_fixture: Callable[[str], str]
    ) -> None:
        # mark_intraday_conflict is audit-only (A10): a failed flag write must
        # be logged but NOT counted as a completion-blocking failure.
        adapter, mocks = _build(
            config, load_fixture, matches_fixture="intraday_conflict.html"
        )
        mocks["matches"].mark_intraday_conflict.side_effect = RuntimeError("audit db down")
        result = adapter.fetch()
        assert result.matches_written == 2     # ingest itself succeeded
        assert result.failures == 0            # audit failure is decoupled
        assert result.complete
        assert _watermark_cursor(mocks)["status"] == "complete"
        assert result.conflicts_flagged == 0   # none successfully flagged
