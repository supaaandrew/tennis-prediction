"""Tests for the Rankings feature family (R4) — `features/rankings.py`.

Exercises the pure helper `_rank_pre_and_stale` and the `RankingsExtractor`:
fresh rank used, the staleness boundary (`max_ranking_staleness_days` fresh /
+1 stale), missing ranking → NULL but NOT stale, `rank_diff = p1 − p2` with NULL
propagation (missing or stale side), the always-present `*_stale` boolean, native
dtypes, and the §M7 lockstep round-trip against the seeded `feature_specs` rows.

All in-memory; no Docker.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tennis.agents.research.context import FeatureContext
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.rankings import (
    RANKINGS_FAMILY,
    RANKINGS_FEATURE_KEYS,
    RankingsExtractor,
    _rank_pre_and_stale,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.storage.postgres.rows import MatchRow, PlayerRankingRow

_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
_AS_OF_DATE = _AS_OF.date()


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


class _FakeRankingRepo:
    """In-memory PlayerRankingRepository keyed by player_id (one row each)."""

    def __init__(self, rows: dict[int, PlayerRankingRow]) -> None:
        self._rows = rows

    def latest_before(
        self, *, player_id: int, on_or_before: date
    ) -> PlayerRankingRow | None:
        row = self._rows.get(player_id)
        if row is None or row.ranking_date > on_or_before:
            return None
        return row


def _rank_row(player_id: int, *, on: date, rank: int) -> PlayerRankingRow:
    return PlayerRankingRow(player_id=player_id, ranking_date=on, rank=rank)


def _current(p1: int = 1, p2: int = 2) -> MatchRow:
    return MatchRow(
        match_id=999,
        tournament_id=900,
        round="R32",
        match_date=_AS_OF_DATE,
        p1_id=p1,
        p2_id=p2,
        status="final",
        source="sackmann",
        source_uid="uid-999",
    )


def _fctx(match: MatchRow) -> FeatureContext:
    return FeatureContext(
        match=match,
        as_of_ts=_AS_OF,
        feature_set="v1",
        surface="Hard",
        indoor=False,
        venue_id=None,
        tier="ATP500",
    )


def _extractor(rows: dict[int, PlayerRankingRow], config: AppConfig) -> RankingsExtractor:
    return RankingsExtractor(ranking_repo=_FakeRankingRepo(rows), config=config)


# ---------------------------------------------------------------------------
class TestRankPreAndStale:
    def test_missing_row_is_null_and_not_stale(self) -> None:
        assert _rank_pre_and_stale(None, as_of_date=_AS_OF_DATE, max_staleness_days=7) == (
            None,
            False,
        )

    def test_within_window_uses_rank(self) -> None:
        row = _rank_row(1, on=_AS_OF_DATE - timedelta(days=3), rank=5)
        assert _rank_pre_and_stale(
            row, as_of_date=_AS_OF_DATE, max_staleness_days=7
        ) == (5, False)

    def test_boundary_equal_to_window_is_fresh(self) -> None:
        row = _rank_row(1, on=_AS_OF_DATE - timedelta(days=7), rank=5)
        assert _rank_pre_and_stale(
            row, as_of_date=_AS_OF_DATE, max_staleness_days=7
        ) == (5, False)

    def test_one_day_past_window_is_stale(self) -> None:
        row = _rank_row(1, on=_AS_OF_DATE - timedelta(days=8), rank=5)
        assert _rank_pre_and_stale(
            row, as_of_date=_AS_OF_DATE, max_staleness_days=7
        ) == (None, True)


class TestRankingsExtractor:
    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ex = _extractor({}, config)
        assert isinstance(ex, FeatureExtractor)
        assert ex.name == RANKINGS_FAMILY

    def test_feature_keys_equal_seeded_rows(self, config: AppConfig) -> None:
        ex = _extractor({}, config)
        seeded = {row.feature_key for row in _REGISTRY["rankings"]}
        assert set(ex.feature_keys()) == seeded
        assert set(RANKINGS_FEATURE_KEYS) == seeded

    def test_fresh_ranks_and_diff(self, config: AppConfig) -> None:
        max_s = config.feature_engineering.max_ranking_staleness_days
        rows = {
            1: _rank_row(1, on=_AS_OF_DATE - timedelta(days=max_s), rank=3),
            2: _rank_row(2, on=_AS_OF_DATE - timedelta(days=1), rank=10),
        }
        out = _extractor(rows, config).extract(_fctx(_current()))
        assert out["p1_rank_pre"] == 3
        assert out["p2_rank_pre"] == 10
        assert out["rank_diff"] == -7
        assert out["p1_rank_stale"] is False
        assert out["p2_rank_stale"] is False

    def test_stale_rank_nulls_value_and_flags_stale(self, config: AppConfig) -> None:
        max_s = config.feature_engineering.max_ranking_staleness_days
        rows = {1: _rank_row(1, on=_AS_OF_DATE - timedelta(days=max_s + 1), rank=3)}
        out = _extractor(rows, config).extract(_fctx(_current()))
        assert out["p1_rank_pre"] is None
        assert out["p1_rank_stale"] is True

    def test_missing_ranking_is_null_and_not_stale(self, config: AppConfig) -> None:
        out = _extractor({}, config).extract(_fctx(_current()))
        assert out["p1_rank_pre"] is None
        assert out["p1_rank_stale"] is False
        assert out["p2_rank_pre"] is None
        assert out["p2_rank_stale"] is False

    def test_rank_diff_null_when_one_side_missing(self, config: AppConfig) -> None:
        rows = {1: _rank_row(1, on=_AS_OF_DATE - timedelta(days=1), rank=3)}
        out = _extractor(rows, config).extract(_fctx(_current()))
        assert out["p1_rank_pre"] == 3
        assert out["p2_rank_pre"] is None
        assert out["rank_diff"] is None

    def test_rank_diff_null_when_one_side_stale(self, config: AppConfig) -> None:
        max_s = config.feature_engineering.max_ranking_staleness_days
        rows = {
            1: _rank_row(1, on=_AS_OF_DATE - timedelta(days=1), rank=3),
            2: _rank_row(2, on=_AS_OF_DATE - timedelta(days=max_s + 1), rank=10),
        }
        out = _extractor(rows, config).extract(_fctx(_current()))
        assert out["p1_rank_pre"] == 3
        assert out["p2_rank_pre"] is None
        assert out["p2_rank_stale"] is True
        assert out["rank_diff"] is None

    def test_dtypes_native_python(self, config: AppConfig) -> None:
        rows = {
            1: _rank_row(1, on=_AS_OF_DATE - timedelta(days=1), rank=3),
            2: _rank_row(2, on=_AS_OF_DATE - timedelta(days=1), rank=10),
        }
        out = _extractor(rows, config).extract(_fctx(_current()))
        assert isinstance(out["p1_rank_pre"], int)
        assert isinstance(out["rank_diff"], int)
        assert isinstance(out["p1_rank_stale"], bool)
