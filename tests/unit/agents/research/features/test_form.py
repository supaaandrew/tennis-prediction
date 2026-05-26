"""Tests for the Form feature family (R4) — `agents/research/features/form.py`.

Exercises the pure helpers (`_counts_as_played`, `_window_record`,
`form_feature_keys`), then the `FormExtractor`: the half-open window bound, the
sparse→NULL threshold at `min_window_samples.elo_form` (M-c), C14 counting
(retirement counted / walkover excluded), the always-present integer
denominator, the `p1 − p2` diff with NULL propagation, window independence, and
the §M7 lockstep round-trip against the seeded `feature_specs` rows.

All in-memory; no Docker. `pit_cut` is not used — the extractor reads
`fctx.as_of_ts` directly — so there is no clock.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.form import (
    FORM_FAMILY,
    FormExtractor,
    _counts_as_played,
    _window_record,
    form_feature_keys,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.storage.postgres.rows import MatchRow

_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


def _match(
    *,
    match_id: int,
    p1_id: int,
    p2_id: int,
    match_date: date,
    winner_id: int | None = None,
    start_ts: datetime | None = None,
    retired: bool = False,
    walkover: bool = False,
    tournament_id: int = 900,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        tournament_id=tournament_id,
        round="R32",
        match_date=match_date,
        p1_id=p1_id,
        p2_id=p2_id,
        status="final",
        source="sackmann",
        source_uid=f"uid-{match_id}",
        start_ts=start_ts,
        winner_id=winner_id,
        retired=retired,
        walkover=walkover,
    )


def _played(
    match_id: int,
    *,
    player: int,
    opp: int,
    on: date,
    won: bool,
    retired: bool = False,
    walkover: bool = False,
) -> MatchRow:
    """A prior match `player` played against `opp` on `on`, with the result."""
    return _match(
        match_id=match_id,
        p1_id=player,
        p2_id=opp,
        match_date=on,
        winner_id=player if won else opp,
        retired=retired,
        walkover=walkover,
    )


def _fctx(match: MatchRow, as_of: datetime, *, surface: str = "Hard") -> FeatureContext:
    return FeatureContext(
        match=match,
        as_of_ts=as_of,
        feature_set="v1",
        surface=surface,  # type: ignore[arg-type]
        indoor=False,
        venue_id=None,
        tier="ATP500",
    )


def _current(p1: int = 1, p2: int = 2) -> MatchRow:
    return _match(match_id=999, p1_id=p1, p2_id=p2, match_date=_AS_OF.date())


# ---------------------------------------------------------------------------
class TestCountsAsPlayed:
    def test_normal_match_with_winner_counts(self) -> None:
        m = _played(1, player=1, opp=2, on=_AS_OF.date(), won=True)
        assert _counts_as_played(m, retirement_counts=True, walkover_counts=False)

    def test_walkover_excluded_when_flag_false(self) -> None:
        m = _played(1, player=1, opp=2, on=_AS_OF.date(), won=True, walkover=True)
        assert not _counts_as_played(m, retirement_counts=True, walkover_counts=False)

    def test_walkover_counted_when_flag_true(self) -> None:
        m = _played(1, player=1, opp=2, on=_AS_OF.date(), won=True, walkover=True)
        assert _counts_as_played(m, retirement_counts=True, walkover_counts=True)

    def test_retirement_counted_when_flag_true(self) -> None:
        m = _played(1, player=1, opp=2, on=_AS_OF.date(), won=False, retired=True)
        assert _counts_as_played(m, retirement_counts=True, walkover_counts=False)

    def test_retirement_excluded_when_flag_false(self) -> None:
        m = _played(1, player=1, opp=2, on=_AS_OF.date(), won=False, retired=True)
        assert not _counts_as_played(m, retirement_counts=False, walkover_counts=False)

    def test_undeterminable_winner_excluded(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=_AS_OF.date(), winner_id=None)
        assert not _counts_as_played(m, retirement_counts=True, walkover_counts=False)


class TestWindowRecord:
    def test_counts_played_and_wins_inside_window(self) -> None:
        on = (_AS_OF - timedelta(days=2)).date()
        matches = (
            _played(1, player=1, opp=10, on=on, won=True),
            _played(2, player=1, opp=11, on=on, won=False),
            _played(3, player=1, opp=12, on=on, won=True),
        )
        played, wins = _window_record(
            matches,
            player_id=1,
            lower=_AS_OF - timedelta(days=7),
            upper=_AS_OF,
            retirement_counts=True,
            walkover_counts=False,
        )
        assert (played, wins) == (3, 2)

    def test_excludes_matches_before_lower_bound(self) -> None:
        matches = (_played(1, player=1, opp=10, on=(_AS_OF - timedelta(days=40)).date(), won=True),)
        played, wins = _window_record(
            matches,
            player_id=1,
            lower=_AS_OF - timedelta(days=7),
            upper=_AS_OF,
            retirement_counts=True,
            walkover_counts=False,
        )
        assert (played, wins) == (0, 0)


class TestFormFeatureKeys:
    def test_shape_and_order(self) -> None:
        assert form_feature_keys((7, 30)) == (
            "p1_win_rate_7d",
            "p2_win_rate_7d",
            "p1_matches_played_7d",
            "p2_matches_played_7d",
            "win_rate_diff_7d",
            "p1_win_rate_30d",
            "p2_win_rate_30d",
            "p1_matches_played_30d",
            "p2_matches_played_30d",
            "win_rate_diff_30d",
        )


class TestFormExtractor:
    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ex = FormExtractor(history=MatchHistoryIndex.build([]), config=config)
        assert isinstance(ex, FeatureExtractor)
        assert ex.name == FORM_FAMILY

    def test_feature_keys_equal_seeded_form_rows(self, config: AppConfig) -> None:
        ex = FormExtractor(history=MatchHistoryIndex.build([]), config=config)
        seeded = {row.feature_key for row in _REGISTRY["form"]}
        assert set(ex.feature_keys()) == seeded

    def test_no_matches_gives_zero_played_and_null_rate(self, config: AppConfig) -> None:
        ex = FormExtractor(history=MatchHistoryIndex.build([]), config=config)
        w = config.features.windows_days[0]

        out = ex.extract(_fctx(_current(), _AS_OF))

        assert out[f"p1_matches_played_{w}d"] == 0
        assert isinstance(out[f"p1_matches_played_{w}d"], int)
        assert out[f"p1_win_rate_{w}d"] is None
        assert out[f"win_rate_diff_{w}d"] is None

    def test_win_rate_null_below_threshold_value_at_threshold(
        self, config: AppConfig
    ) -> None:
        thr = config.feature_engineering.min_window_samples.elo_form
        w = config.features.windows_days[0]
        on = (_AS_OF - timedelta(days=1)).date()
        matches = [
            _played(i, player=1, opp=100 + i, on=on, won=(i < 3)) for i in range(thr)
        ]

        at = FormExtractor(history=MatchHistoryIndex.build(matches), config=config)
        out_at = at.extract(_fctx(_current(), _AS_OF))
        assert out_at[f"p1_matches_played_{w}d"] == thr
        assert out_at[f"p1_win_rate_{w}d"] == pytest.approx(3 / thr)

        below = FormExtractor(
            history=MatchHistoryIndex.build(matches[:-1]), config=config
        )
        out_below = below.extract(_fctx(_current(), _AS_OF))
        assert out_below[f"p1_matches_played_{w}d"] == thr - 1
        assert out_below[f"p1_win_rate_{w}d"] is None

    def test_window_lower_bound_inclusive_and_older_excluded(
        self, config: AppConfig
    ) -> None:
        w = config.features.windows_days[0]
        on_lower = (_AS_OF - timedelta(days=w)).date()  # instant == lower → included
        older = (_AS_OF - timedelta(days=w + 1)).date()  # < lower → excluded
        matches = [
            _played(1, player=1, opp=100, on=on_lower, won=True),
            _played(2, player=1, opp=101, on=older, won=True),
        ]

        out = FormExtractor(
            history=MatchHistoryIndex.build(matches), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert out[f"p1_matches_played_{w}d"] == 1

    def test_match_at_as_of_is_excluded_upper_bound(self, config: AppConfig) -> None:
        w = config.features.windows_days[0]
        # A scheduled prior match whose instant equals as_of must not count (PIT).
        at_cut = _match(
            match_id=1, p1_id=1, p2_id=100, match_date=_AS_OF.date(),
            winner_id=1, start_ts=_AS_OF,
        )

        out = FormExtractor(
            history=MatchHistoryIndex.build([at_cut]), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert out[f"p1_matches_played_{w}d"] == 0

    def test_walkover_excluded_retirement_counted(self, config: AppConfig) -> None:
        w = config.features.windows_days[0]
        on = (_AS_OF - timedelta(days=1)).date()
        matches = [
            _played(1, player=1, opp=100, on=on, won=True),
            _played(2, player=1, opp=101, on=on, won=False, retired=True),
            _played(3, player=1, opp=102, on=on, won=True, walkover=True),
        ]

        out = FormExtractor(
            history=MatchHistoryIndex.build(matches), config=config
        ).extract(_fctx(_current(), _AS_OF))

        # 2 counted (win + retirement-loss); walkover excluded.
        assert out[f"p1_matches_played_{w}d"] == 2

    def test_player_as_p2_in_prior_match_is_counted(self, config: AppConfig) -> None:
        w = config.features.windows_days[0]
        on = (_AS_OF - timedelta(days=1)).date()
        # Player 1 appears as p2_id and won.
        prior = _match(match_id=1, p1_id=100, p2_id=1, match_date=on, winner_id=1)

        out = FormExtractor(
            history=MatchHistoryIndex.build([prior]), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert out[f"p1_matches_played_{w}d"] == 1

    def test_win_rate_diff_is_p1_minus_p2(self, config: AppConfig) -> None:
        thr = config.feature_engineering.min_window_samples.elo_form
        w = config.features.windows_days[0]
        on = (_AS_OF - timedelta(days=1)).date()
        p1_matches = [_played(i, player=1, opp=100 + i, on=on, won=True) for i in range(thr)]
        p2_matches = [
            _played(500 + i, player=2, opp=200 + i, on=on, won=False) for i in range(thr)
        ]

        out = FormExtractor(
            history=MatchHistoryIndex.build(p1_matches + p2_matches), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert out[f"p1_win_rate_{w}d"] == 1.0
        assert out[f"p2_win_rate_{w}d"] == 0.0
        assert out[f"win_rate_diff_{w}d"] == 1.0

    def test_win_rate_diff_null_when_one_side_sparse(self, config: AppConfig) -> None:
        thr = config.feature_engineering.min_window_samples.elo_form
        w = config.features.windows_days[0]
        on = (_AS_OF - timedelta(days=1)).date()
        p1_matches = [_played(i, player=1, opp=100 + i, on=on, won=True) for i in range(thr)]
        p2_matches = [_played(900, player=2, opp=200, on=on, won=True)]  # 1 < thr

        out = FormExtractor(
            history=MatchHistoryIndex.build(p1_matches + p2_matches), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert out[f"p1_win_rate_{w}d"] is not None
        assert out[f"p2_win_rate_{w}d"] is None
        assert out[f"win_rate_diff_{w}d"] is None

    def test_windows_are_independent(self, config: AppConfig) -> None:
        thr = config.feature_engineering.min_window_samples.elo_form
        # 20 days ago: inside 30d/90d/365d, outside 7d/14d.
        on = (_AS_OF - timedelta(days=20)).date()
        matches = [_played(i, player=1, opp=100 + i, on=on, won=True) for i in range(thr)]

        out = FormExtractor(
            history=MatchHistoryIndex.build(matches), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert out["p1_matches_played_7d"] == 0
        assert out["p1_matches_played_14d"] == 0
        assert out["p1_matches_played_30d"] == thr

    def test_dtypes_are_native_python(self, config: AppConfig) -> None:
        thr = config.feature_engineering.min_window_samples.elo_form
        w = config.features.windows_days[0]
        on = (_AS_OF - timedelta(days=1)).date()
        # Both sides need >= thr matches so every value is non-NULL.
        p1_matches = [_played(i, player=1, opp=100 + i, on=on, won=(i < 3)) for i in range(thr)]
        p2_matches = [_played(500 + i, player=2, opp=200 + i, on=on, won=True) for i in range(thr)]

        out = FormExtractor(
            history=MatchHistoryIndex.build(p1_matches + p2_matches), config=config
        ).extract(_fctx(_current(), _AS_OF))

        assert isinstance(out[f"p1_win_rate_{w}d"], float)
        assert isinstance(out[f"p1_matches_played_{w}d"], int)
        # Diff is float here (both sides have rates).
        assert isinstance(out[f"win_rate_diff_{w}d"], float)
