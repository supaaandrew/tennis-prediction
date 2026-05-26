"""Tests for the Serve/return family (R5a) — `features/serve_return.py`.

Exercises the pure aggregation helpers (`_PairSum`, `_ServeAggregate`) and the
`ServeReturnExtractor`: each §15.5 ratio over career and trailing-365d windows,
the min-sample (10) NULL gate, the pre-1991 / absent-stats NULL path, the
paired-presence summation (a missing field is treated as absent, never coerced
to 0), zero-denominator NULLs (`bp_save` with no break points, `second_serve`
with no second serves), the career-vs-365d window split, the dominance diff and
its NULL propagation, and the §M7 lockstep round-trip.

All in-memory; no Docker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.serve_return import (
    SERVE_RETURN_FAMILY,
    SERVE_RETURN_FEATURE_KEYS,
    ServeReturnExtractor,
    _PairSum,
    _ServeAggregate,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.storage.postgres.rows import MatchRow, MatchStatRow

_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
_P1 = 10
_P2 = 20
_OPP = 99

# Default per-match serve line: yields first_serve_pct=0.625, first_serve_win=0.8,
# second_serve_win=0.5, ace_rate=0.1, df_rate=0.0375, bp_save=0.625.
_DEFAULT_STAT: dict[str, Any] = dict(
    serve_pts=80,
    first_in=50,
    first_won=40,
    second_won=15,
    aces=8,
    double_faults=3,
    bp_saved=5,
    bp_faced=8,
)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


class _FakeStatRepo:
    """In-memory MatchStatRepository keyed on (match_id, player_id)."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, int], MatchStatRow] = {}

    def add(self, stat: MatchStatRow) -> None:
        self._store[(stat.match_id, stat.player_id)] = stat

    def get(self, *, match_id: int, player_id: int) -> MatchStatRow | None:
        return self._store.get((match_id, player_id))


def _match(match_id: int, player_id: int, days_ago: int) -> MatchRow:
    inst = _AS_OF - timedelta(days=days_ago)
    return MatchRow(
        match_id=match_id,
        tournament_id=900,
        round="R32",
        match_date=inst.date(),
        p1_id=player_id,
        p2_id=_OPP,
        status="final",
        source="sackmann",
        source_uid=f"uid-{match_id}",
        start_ts=inst,
        winner_id=player_id,
    )


def _stat(match_id: int, player_id: int, **overrides: Any) -> MatchStatRow:
    fields = {**_DEFAULT_STAT, **overrides}
    return MatchStatRow(match_id=match_id, player_id=player_id, is_winner=True, **fields)


def _current() -> MatchRow:
    return MatchRow(
        match_id=999,
        tournament_id=900,
        round="R32",
        match_date=_AS_OF.date(),
        p1_id=_P1,
        p2_id=_P2,
        status="scheduled",
        source="sackmann",
        source_uid="uid-999",
        start_ts=_AS_OF,
    )


def _fctx() -> FeatureContext:
    return FeatureContext(
        match=_current(),
        as_of_ts=_AS_OF,
        feature_set="v1",
        surface="Hard",
        indoor=False,
        venue_id=None,
        tier="ATP500",
    )


def _build(
    config: AppConfig,
    *,
    p1: list[tuple[int, dict[str, Any] | None]] | None = None,
    p2: list[tuple[int, dict[str, Any] | None]] | None = None,
) -> ServeReturnExtractor:
    """Build the extractor from per-player `(days_ago, stat_overrides)` entries.
    A `None` stat means the match has no `match_stats` row (repo returns None)."""
    matches: list[MatchRow] = []
    repo = _FakeStatRepo()
    mid = 0
    for entries, pid in ((p1 or [], _P1), (p2 or [], _P2)):
        for days_ago, overrides in entries:
            mid += 1
            matches.append(_match(mid, pid, days_ago))
            if overrides is not None:
                repo.add(_stat(mid, pid, **overrides))
    return ServeReturnExtractor(
        history=MatchHistoryIndex.build(matches),
        stat_repo=repo,
        config=config,
    )


def _entries(n: int, *, start_day: int = 10, **overrides: Any) -> list[tuple[int, dict[str, Any]]]:
    """`n` in-window matches (within 365d) carrying the given stat overrides."""
    return [(start_day + i, dict(overrides)) for i in range(n)]


# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_pairsum_ratio(self) -> None:
        p = _PairSum()
        p.add(50, 80)
        p.add(50, 80)
        assert p.ratio() == pytest.approx(100 / 160)

    def test_pairsum_null_on_zero_denominator(self) -> None:
        assert _PairSum().ratio() is None

    def test_pairsum_skips_when_either_missing(self) -> None:
        p = _PairSum()
        p.add(None, 80)  # numerator absent
        p.add(40, None)  # denominator absent
        assert p.ratio() is None

    def test_aggregate_ignores_zero_serve_pts(self) -> None:
        agg = _ServeAggregate()
        agg.add(MatchStatRow(match_id=1, player_id=_P1, is_winner=True, serve_pts=0))
        agg.add(MatchStatRow(match_id=2, player_id=_P1, is_winner=True, serve_pts=None))
        assert agg.samples == 0

    def test_aggregate_counts_sample_and_sums(self) -> None:
        agg = _ServeAggregate()
        agg.add(MatchStatRow(match_id=1, player_id=_P1, is_winner=True, **_DEFAULT_STAT))
        assert agg.samples == 1
        assert agg.first_serve.ratio() == pytest.approx(0.625)

    def test_aggregate_excludes_corrupt_second_serve_denominator(self) -> None:
        # A corrupt row (first_in > serve_pts) would give a negative second-serve
        # denominator; it must be treated as absent so it never skews the ratio.
        agg = _ServeAggregate()
        agg.add(MatchStatRow(match_id=1, player_id=_P1, is_winner=True, **_DEFAULT_STAT))
        agg.add(
            MatchStatRow(
                match_id=2, player_id=_P1, is_winner=True,
                serve_pts=80, first_in=90, second_won=5,  # first_in > serve_pts
            )
        )
        assert agg.samples == 2  # both have serve_pts > 0
        # Only the valid row feeds the second-serve ratio: 15 / (80 − 50) = 0.5.
        assert agg.second_serve_win.ratio() == pytest.approx(0.5)


class TestServeReturnContract:
    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ex = _build(config)
        assert isinstance(ex, FeatureExtractor)
        assert ex.name == SERVE_RETURN_FAMILY

    def test_feature_keys_equal_seeded_rows(self, config: AppConfig) -> None:
        ex = _build(config)
        seeded = {row.feature_key for row in _REGISTRY["serve_return"]}
        assert set(ex.feature_keys()) == seeded
        assert set(SERVE_RETURN_FEATURE_KEYS) == seeded

    def test_all_keys_present_in_output(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10), p2=_entries(10)).extract(_fctx())
        assert set(out) == set(SERVE_RETURN_FEATURE_KEYS)

    def test_dtypes_native_float(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10), p2=_entries(10)).extract(_fctx())
        assert isinstance(out["p1_first_serve_pct_career"], float)
        assert isinstance(out["serve_dominance_diff_365d"], float)


class TestServeReturnNullGates:
    def test_no_stats_all_null(self, config: AppConfig) -> None:
        # Matches exist but carry no match_stats row (pre-1991 analogue).
        out = _build(config, p1=[(10 + i, None) for i in range(12)]).extract(_fctx())
        assert all(out[k] is None for k in SERVE_RETURN_FEATURE_KEYS)

    def test_below_min_sample_null(self, config: AppConfig) -> None:
        # 9 usable samples < min (10) → NULL.
        out = _build(config, p1=_entries(9)).extract(_fctx())
        assert out["p1_first_serve_pct_career"] is None
        assert out["p1_first_serve_pct_365d"] is None

    def test_at_min_sample_defined(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)

    def test_serve_pts_zero_not_a_sample(self, config: AppConfig) -> None:
        # 10 matches but each has serve_pts=0 → no usable samples → NULL.
        out = _build(config, p1=_entries(10, serve_pts=0)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] is None


class TestServeReturnRatios:
    def test_each_ratio(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)
        assert out["p1_first_serve_win_pct_365d"] == pytest.approx(0.8)
        assert out["p1_second_serve_win_pct_365d"] == pytest.approx(0.5)
        assert out["p1_ace_rate_365d"] == pytest.approx(0.1)
        assert out["p1_df_rate_365d"] == pytest.approx(0.0375)
        assert out["p1_bp_save_pct_365d"] == pytest.approx(0.625)

    def test_bp_save_null_on_zero_bp_faced(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10, bp_saved=0, bp_faced=0)).extract(_fctx())
        assert out["p1_bp_save_pct_365d"] is None
        # Other ratios still defined.
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)

    def test_second_serve_null_when_no_second_serves(self, config: AppConfig) -> None:
        # serve_pts == first_in → second-serve denominator is 0 → NULL.
        out = _build(config, p1=_entries(10, serve_pts=50, first_in=50)).extract(_fctx())
        assert out["p1_second_serve_win_pct_365d"] is None

    def test_paired_presence_missing_field_treated_absent(self, config: AppConfig) -> None:
        # first_in absent: first_serve_pct cannot be computed (NULL), but ace_rate
        # (aces / serve_pts) still aggregates, and the matches still count as
        # samples (serve_pts present).
        out = _build(config, p1=_entries(10, first_in=None)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] is None
        assert out["p1_ace_rate_365d"] == pytest.approx(0.1)


class TestServeReturnWindows:
    def test_career_vs_365d_split(self, config: AppConfig) -> None:
        # 10 recent (first_in=50/serve_pts=80 → 0.625) + 10 older-than-365d
        # (first_in=80/serve_pts=80 → 1.0). Career blends both; 365d is recent-only.
        recent = _entries(10, first_in=50)
        old = [(400 + i, dict(first_in=80)) for i in range(10)]
        out = _build(config, p1=recent + old).extract(_fctx())
        # career: (10*50 + 10*80) / (20*80) = 1300/1600
        assert out["p1_first_serve_pct_career"] == pytest.approx(1300 / 1600)
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)

    def test_old_only_recent_null_career_defined(self, config: AppConfig) -> None:
        # All samples older than 365d → 365d window empty (NULL), career defined.
        out = _build(config, p1=[(400 + i, {}) for i in range(10)]).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] is None
        assert out["p1_first_serve_pct_career"] == pytest.approx(0.625)


class TestServeDominanceDiff:
    def test_diff_is_p1_minus_p2(self, config: AppConfig) -> None:
        # p1 first_serve_win = 40/50 = 0.8; p2 = 30/50 = 0.6 → diff 0.2.
        out = _build(
            config,
            p1=_entries(10),
            p2=_entries(10, first_won=30),
        ).extract(_fctx())
        assert out["p1_first_serve_win_pct_365d"] == pytest.approx(0.8)
        assert out["p2_first_serve_win_pct_365d"] == pytest.approx(0.6)
        assert out["serve_dominance_diff_365d"] == pytest.approx(0.2)

    def test_diff_null_when_one_side_sparse(self, config: AppConfig) -> None:
        # p2 has < 10 samples → its 365d win pct NULL → diff NULL.
        out = _build(config, p1=_entries(10), p2=_entries(5)).extract(_fctx())
        assert out["p2_first_serve_win_pct_365d"] is None
        assert out["serve_dominance_diff_365d"] is None
