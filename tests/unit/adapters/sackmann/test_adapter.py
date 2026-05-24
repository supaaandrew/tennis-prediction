"""Unit tests for the Sackmann adapter orchestration.

All repositories and the filesystem reader are mocked; the resolver is a
MagicMock returning deterministic player_ids so the tests exercise the
adapter's control flow (staleness, watermarks, skip/dead-letter routing,
intraday-conflict pass) rather than identity resolution.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sqlalchemy.exc import OperationalError

from tennis.adapters.sackmann.adapter import SackmannAdapter
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import PlayerResolutionError, SackmannStalenessError

NOW = datetime(2026, 5, 23, 6, 30, tzinfo=UTC)


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    return load_config(config_path)


def _match_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "tourney_id": "2023-580",
        "tourney_name": "Australian Open",
        "tourney_level": "G",
        "surface": "Hard",
        "draw_size": "128",
        "tourney_date": "20230116",
        "match_num": "101",
        "round": "R128",
        "score": "6-4 6-3 6-2",
        "best_of": "5",
        "minutes": "142",
        "winner_id": "104925",
        "winner_name": "Novak Djokovic",
        "winner_ioc": "SRB",
        "winner_hand": "R",
        "winner_rank": "5",
        "winner_rank_points": "3000",
        "loser_id": "200000",
        "loser_name": "Some Player",
        "loser_ioc": "USA",
        "loser_hand": "L",
        "loser_rank": "100",
        "loser_rank_points": "600",
    }
    base.update(overrides)
    return base


def _build(config: AppConfig, *, resolve_side_effect: object = None):
    reader = MagicMock()
    reader.dir_mtime.return_value = NOW - timedelta(hours=1)
    reader.head_commit_sha.return_value = None
    reader.read_players.return_value = []
    reader.read_matches.return_value = []
    reader.read_rankings.return_value = []

    resolver = MagicMock()
    if resolve_side_effect is not None:
        resolver.resolve.side_effect = resolve_side_effect

    mocks = {
        "reader": reader,
        "resolver": resolver,
        "players": MagicMock(),
        "rankings": MagicMock(),
        "tournaments": MagicMock(),
        "matches": MagicMock(),
        "match_stats": MagicMock(),
        "watermarks": MagicMock(),
        "dead_letter": MagicMock(),
    }
    mocks["watermarks"].get.return_value = None

    adapter = SackmannAdapter(
        config=config,
        clock=FrozenClock(NOW),
        reader=reader,
        resolver=resolver,
        players=mocks["players"],
        rankings=mocks["rankings"],
        tournaments=mocks["tournaments"],
        matches=mocks["matches"],
        match_stats=mocks["match_stats"],
        watermarks=mocks["watermarks"],
        dead_letter=mocks["dead_letter"],
    )
    return adapter, mocks


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
def test_staleness_raises_when_mirror_too_old(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["reader"].dir_mtime.return_value = NOW - timedelta(days=10)
    with pytest.raises(SackmannStalenessError):
        adapter.check_staleness()


def test_staleness_passes_when_fresh(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["reader"].dir_mtime.return_value = NOW - timedelta(days=1)
    adapter.check_staleness()  # must not raise


# ---------------------------------------------------------------------------
# Watermark gating
# ---------------------------------------------------------------------------
def test_watermark_read_before_season_ingest(config: AppConfig) -> None:
    adapter, mocks = _build(config, resolve_side_effect=[100, 200])
    mocks["reader"].read_matches.return_value = [_match_row()]

    adapter.ingest_season(2023)

    mocks["watermarks"].get.assert_called_once_with(source="sackmann", scope="2023")


def test_watermark_updated_after_season_completes(config: AppConfig) -> None:
    adapter, mocks = _build(config, resolve_side_effect=[100, 200])
    mocks["reader"].read_matches.return_value = [_match_row()]

    adapter.ingest_season(2023)

    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.source == "sackmann"
    assert row.scope == "2023"
    assert row.cursor["status"] == "complete"


def test_completed_season_is_skipped(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["watermarks"].get.return_value = MagicMock(cursor={"status": "complete"})

    result = adapter.ingest_season(2023)

    assert result.skipped is True
    mocks["reader"].read_matches.assert_not_called()


# ---------------------------------------------------------------------------
# Row routing
# ---------------------------------------------------------------------------
def test_excluded_tier_silently_skipped(config: AppConfig) -> None:
    # tourney_level 'C' -> tier 'Other' -> not in included_tiers.
    adapter, mocks = _build(config, resolve_side_effect=[1, 2])
    mocks["reader"].read_matches.return_value = [_match_row(tourney_level="C", draw_size="32")]

    result = adapter.ingest_season(2023)

    assert result.matches_ingested == 0
    mocks["matches"].upsert.assert_not_called()
    mocks["dead_letter"].append.assert_not_called()


def test_qualifying_round_silently_skipped(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["reader"].read_matches.return_value = [_match_row(round="Q1")]

    result = adapter.ingest_season(2023)

    assert result.matches_ingested == 0
    mocks["matches"].upsert.assert_not_called()
    mocks["dead_letter"].append.assert_not_called()


def test_unresolvable_player_dead_lettered_and_continues(config: AppConfig) -> None:
    # Row 1 winner raises; row 2 resolves cleanly.
    adapter, mocks = _build(
        config, resolve_side_effect=[PlayerResolutionError("nope"), 100, 200]
    )
    mocks["reader"].read_matches.return_value = [
        _match_row(match_num="1"),
        _match_row(match_num="2", round="R64"),
    ]

    result = adapter.ingest_season(2023)

    assert result.matches_ingested == 1
    mocks["dead_letter"].append.assert_called_once()
    assert mocks["matches"].upsert.call_count == 1


def test_unexpected_exception_dead_lettered_and_continues(config: AppConfig) -> None:
    adapter, mocks = _build(config, resolve_side_effect=[ValueError("boom"), 100, 200])
    mocks["reader"].read_matches.return_value = [
        _match_row(match_num="1"),
        _match_row(match_num="2", round="R64"),
    ]

    result = adapter.ingest_season(2023)

    assert result.matches_ingested == 1
    mocks["dead_letter"].append.assert_called_once()
    dead = mocks["dead_letter"].append.call_args.args[0]
    assert dead.error["type"] == "ValueError"


# ---------------------------------------------------------------------------
# Intraday-conflict audit pass — Sackmann rows are tournament_week precision
# ---------------------------------------------------------------------------
def test_sackmann_rows_never_flag_intraday_conflict(config: AppConfig) -> None:
    # Player 100 plays QF and SF of the same tournament -> same tourney_date.
    # That is normal multi-round progression, NOT a same-day conflict, so the
    # audit must never fire for tournament_week-precision Sackmann rows.
    adapter, mocks = _build(config, resolve_side_effect=[100, 200, 100, 300])
    mocks["reader"].read_matches.return_value = [
        _match_row(match_num="1", round="QF"),
        _match_row(match_num="2", round="SF"),
    ]

    result = adapter.ingest_season(2023)

    assert result.matches_ingested == 2
    assert result.conflicts_flagged == 0
    mocks["matches"].mark_intraday_conflict.assert_not_called()


# ---------------------------------------------------------------------------
# Watermark failure-awareness (FIX 1)
# ---------------------------------------------------------------------------
def test_repo_failure_marks_season_incomplete(config: AppConfig) -> None:
    # A storage failure (DB rejected the row) must NOT complete the season.
    adapter, mocks = _build(config, resolve_side_effect=[100, 200])
    mocks["matches"].upsert.side_effect = OperationalError("ingest", {}, Exception("db down"))
    mocks["reader"].read_matches.return_value = [_match_row()]

    result = adapter.ingest_season(2023)

    assert result.repo_failures == 1
    assert result.complete is False
    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.cursor["status"] == "incomplete"
    assert row.cursor["repo_failures"] == 1


def test_validation_only_failures_still_complete_season(config: AppConfig) -> None:
    # All rows dead-lettered for VALIDATION reasons -> season still complete
    # (dead-lettered rows are intentionally excluded, not lost).
    adapter, mocks = _build(config)
    mocks["reader"].read_matches.return_value = [_match_row(surface="Mud")]  # parse raises

    result = adapter.ingest_season(2023)

    assert result.validation_failures == 1
    assert result.repo_failures == 0
    assert result.complete is True
    mocks["dead_letter"].append.assert_called_once()
    row = mocks["watermarks"].upsert.call_args.args[0]
    assert row.cursor["status"] == "complete"


# ---------------------------------------------------------------------------
# rank=0 normalization (FIX 4)
# ---------------------------------------------------------------------------
def test_rank_zero_does_not_block_match_ingest(config: AppConfig) -> None:
    # Sackmann sometimes records rank=0; the schema CHECK(rank > 0) would
    # reject it. The parser normalizes it to None so the ranking is simply
    # skipped and the match + stats still land.
    adapter, mocks = _build(config, resolve_side_effect=[100, 200])
    mocks["reader"].read_matches.return_value = [
        _match_row(winner_rank="0", loser_rank="0")
    ]

    result = adapter.ingest_season(2023)

    assert result.matches_ingested == 1
    mocks["matches"].upsert.assert_called_once()
    assert mocks["match_stats"].upsert.call_count == 2
    mocks["rankings"].upsert.assert_not_called()
    mocks["dead_letter"].append.assert_not_called()


# ---------------------------------------------------------------------------
# Players + rankings ingest
# ---------------------------------------------------------------------------
def test_ingest_players_upserts_and_registers(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["reader"].read_players.return_value = [
        {
            "player_id": "104925",
            "name_first": "Novak",
            "name_last": "Djokovic",
            "hand": "R",
            "dob": "19870522",
            "ioc": "SRB",
            "height": "188",
        }
    ]

    count = adapter.ingest_players()

    assert count == 1
    mocks["players"].upsert.assert_called_once()
    mocks["resolver"].register.assert_called_once()


def test_ingest_rankings_dead_letters_unknown_atp_id(config: AppConfig) -> None:
    adapter, mocks = _build(config)
    mocks["players"].get.return_value = None  # atp_id not in players table
    only_decade = config.sources.sackmann.ranking_decades[0]

    def _read(decade: str):
        if decade == only_decade:
            return [{"ranking_date": "20230102", "rank": "1", "player": "999", "points": "100"}]
        return []

    mocks["reader"].read_rankings.side_effect = _read

    count = adapter.ingest_rankings()

    assert count == 0
    mocks["dead_letter"].append.assert_called_once()
    mocks["rankings"].upsert.assert_not_called()
