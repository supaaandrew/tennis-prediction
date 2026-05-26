"""Tests for the Head-to-head feature family (R4) — `features/h2h.py`.

Exercises the pure helpers (`_rate`, `_confidence_weighted`, `_age_years`,
`_recency_weighted_rate`) and the `H2HExtractor`: base counts/rate, the
p1-perspective win mapping (order-independent), C14 counting (walkover excluded /
retirement counted), the surface filter (resolved / unresolved tournaments), the
§M1 confidence shrink toward 0.5, the §M1 recency decay (years from `as_of`), and
the §M7 lockstep round-trip against the seeded `feature_specs` rows.

All in-memory; no Docker.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.h2h import (
    H2H_FAMILY,
    H2H_FEATURE_KEYS,
    H2HExtractor,
    _age_years,
    _confidence_weighted,
    _rate,
    _recency_weighted_rate,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.storage.postgres.rows import MatchRow, Surface, TournamentRow

_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


class _FakeTournamentRepo:
    """In-memory TournamentRepository: tournament_id → surface."""

    def __init__(self, surfaces: dict[int, Surface]) -> None:
        self._surfaces = surfaces

    def get(self, tournament_id: int) -> TournamentRow | None:
        surface = self._surfaces.get(tournament_id)
        if surface is None:
            return None
        return TournamentRow(
            tournament_id=tournament_id,
            season=2024,
            slug=f"t{tournament_id}",
            name=f"Tournament {tournament_id}",
            tier="ATP500",
            surface=surface,
            indoor=False,
        )


def _repo(mapping: dict[int, Surface] | None = None) -> _FakeTournamentRepo:
    return _FakeTournamentRepo(mapping if mapping is not None else {900: "Hard"})


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


def _meeting(
    match_id: int,
    *,
    winner: int,
    on: date | None = None,
    start_ts: datetime | None = None,
    tournament_id: int = 900,
    retired: bool = False,
    walkover: bool = False,
) -> MatchRow:
    """A prior meeting between players 1 and 2, won by `winner`. Defaults to a
    date strictly before `as_of` (an instant == as_of is excluded by PIT)."""
    return _match(
        match_id=match_id,
        p1_id=1,
        p2_id=2,
        match_date=on or (_AS_OF - timedelta(days=10)).date(),
        winner_id=winner,
        start_ts=start_ts,
        tournament_id=tournament_id,
        retired=retired,
        walkover=walkover,
    )


def _current(p1: int = 1, p2: int = 2) -> MatchRow:
    return _match(match_id=999, p1_id=p1, p2_id=p2, match_date=_AS_OF.date())


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


def _extractor(meetings: list[MatchRow], config: AppConfig, repo=None) -> H2HExtractor:
    return H2HExtractor(
        history=MatchHistoryIndex.build(meetings),
        tournament_repo=repo or _repo(),
        config=config,
    )


# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_rate_null_for_zero_total(self) -> None:
        assert _rate(0, 0) is None

    def test_rate_value(self) -> None:
        assert _rate(3, 4) == 0.75

    def test_confidence_shrinks_below_full_sample(self) -> None:
        # rate 1.0, n=1, full=5 → 0.5 + 0.5*0.2 = 0.6
        assert _confidence_weighted(1.0, 1, 5) == pytest.approx(0.6)

    def test_confidence_equals_rate_at_full_sample(self) -> None:
        assert _confidence_weighted(0.8, 5, 5) == pytest.approx(0.8)

    def test_confidence_caps_above_full_sample(self) -> None:
        assert _confidence_weighted(0.0, 12, 5) == pytest.approx(0.0)

    def test_confidence_below_half_shrinks_up(self) -> None:
        # A losing record shrinks toward 0.5 from below.
        assert _confidence_weighted(0.0, 1, 5) == pytest.approx(0.4)

    def test_age_years_from_as_of(self) -> None:
        two_years = _AS_OF - timedelta(days=730.5)
        assert _age_years(_AS_OF, two_years) == pytest.approx(2.0)

    def test_recency_weighted_equal_ages_equals_simple_rate(self) -> None:
        inst = _AS_OF - timedelta(days=10)
        meetings = (
            _meeting(1, winner=1, start_ts=inst),
            _meeting(2, winner=2, start_ts=inst),
        )
        assert _recency_weighted_rate(
            meetings, p1_id=1, as_of=_AS_OF, halflife_years=2.0
        ) == pytest.approx(0.5)

    def test_recency_weighted_matches_formula(self) -> None:
        win = _meeting(1, winner=1, start_ts=_AS_OF - timedelta(days=30))
        loss = _meeting(2, winner=2, start_ts=_AS_OF - timedelta(days=730))
        result = _recency_weighted_rate(
            (win, loss), p1_id=1, as_of=_AS_OF, halflife_years=2.0
        )
        w_win = math.exp(-(30 / 365.25) / 2.0)
        w_loss = math.exp(-(730 / 365.25) / 2.0)
        assert result == pytest.approx(w_win / (w_win + w_loss))


class TestH2HExtractorBase:
    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ex = _extractor([], config)
        assert isinstance(ex, FeatureExtractor)
        assert ex.name == H2H_FAMILY

    def test_feature_keys_equal_seeded_rows(self, config: AppConfig) -> None:
        ex = _extractor([], config)
        seeded = {row.feature_key for row in _REGISTRY["h2h"]}
        assert set(ex.feature_keys()) == seeded
        assert set(H2H_FEATURE_KEYS) == seeded

    def test_no_meetings_counts_zero_rates_null(self, config: AppConfig) -> None:
        out = _extractor([], config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_matches"] == 0
        assert out["h2h_p1_wins"] == 0
        assert out["h2h_p1_win_rate"] is None
        assert out["h2h_surface_matches"] == 0
        assert out["h2h_surface_p1_win_rate"] is None
        assert out["h2h_win_rate_confidence"] is None
        assert out["h2h_win_rate_weighted"] is None

    def test_counts_wins_and_rate(self, config: AppConfig) -> None:
        meetings = [
            _meeting(1, winner=1),
            _meeting(2, winner=1),
            _meeting(3, winner=1),
            _meeting(4, winner=2),
        ]
        out = _extractor(meetings, config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_matches"] == 4
        assert out["h2h_p1_wins"] == 3
        assert out["h2h_p1_win_rate"] == 0.75

    def test_p1_perspective_is_order_independent(self, config: AppConfig) -> None:
        # Meeting stored with reversed roles (p1_id=2, p2_id=1), winner = player 1.
        reversed_meeting = _match(
            match_id=1,
            p1_id=2,
            p2_id=1,
            match_date=(_AS_OF - timedelta(days=10)).date(),
            winner_id=1,
        )
        out = _extractor([reversed_meeting], config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_matches"] == 1
        assert out["h2h_p1_wins"] == 1
        assert out["h2h_p1_win_rate"] == 1.0

    def test_walkover_excluded_retirement_counted(self, config: AppConfig) -> None:
        meetings = [
            _meeting(1, winner=1),
            _meeting(2, winner=2, retired=True),
            _meeting(3, winner=1, walkover=True),
        ]
        out = _extractor(meetings, config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_matches"] == 2
        assert out["h2h_p1_wins"] == 1

    def test_meeting_at_as_of_excluded(self, config: AppConfig) -> None:
        at_cut = _meeting(1, winner=1, start_ts=_AS_OF)
        out = _extractor([at_cut], config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_matches"] == 0

    def test_dtypes_native_python(self, config: AppConfig) -> None:
        out = _extractor([_meeting(1, winner=1)], config).extract(
            _fctx(_current(), _AS_OF)
        )
        assert isinstance(out["h2h_matches"], int)
        assert isinstance(out["h2h_p1_wins"], int)
        assert isinstance(out["h2h_p1_win_rate"], float)
        assert isinstance(out["h2h_win_rate_confidence"], float)
        assert isinstance(out["h2h_win_rate_weighted"], float)


class TestH2HSurfaceFilter:
    def test_counts_only_matching_surface(self, config: AppConfig) -> None:
        repo = _repo({901: "Hard", 902: "Clay"})
        meetings = [
            _meeting(1, winner=1, tournament_id=901),  # Hard win
            _meeting(2, winner=2, tournament_id=902),  # Clay
            _meeting(3, winner=1, tournament_id=901),  # Hard win
        ]
        out = _extractor(meetings, config, repo).extract(
            _fctx(_current(), _AS_OF, surface="Hard")
        )
        assert out["h2h_matches"] == 3
        assert out["h2h_surface_matches"] == 2
        assert out["h2h_surface_p1_win_rate"] == 1.0

    def test_surface_rate_null_when_no_surface_meetings(self, config: AppConfig) -> None:
        repo = _repo({902: "Clay"})
        meetings = [_meeting(1, winner=1, tournament_id=902)]
        out = _extractor(meetings, config, repo).extract(
            _fctx(_current(), _AS_OF, surface="Hard")
        )
        assert out["h2h_matches"] == 1
        assert out["h2h_surface_matches"] == 0
        assert out["h2h_surface_p1_win_rate"] is None

    def test_unresolved_tournament_excluded_from_surface_kept_in_overall(
        self, config: AppConfig
    ) -> None:
        repo = _repo({})  # tournament 900 unresolved
        meetings = [_meeting(1, winner=1, tournament_id=900)]
        out = _extractor(meetings, config, repo).extract(
            _fctx(_current(), _AS_OF, surface="Hard")
        )
        assert out["h2h_matches"] == 1
        assert out["h2h_surface_matches"] == 0


class TestH2HAdvanced:
    def test_confidence_shrinks_toward_half(self, config: AppConfig) -> None:
        full = config.features.h2h.confidence_full_sample
        out = _extractor([_meeting(1, winner=1)], config).extract(
            _fctx(_current(), _AS_OF)
        )
        assert out["h2h_p1_win_rate"] == 1.0
        assert out["h2h_win_rate_confidence"] == pytest.approx(
            0.5 + 0.5 * min(1 / full, 1.0)
        )

    def test_confidence_equals_rate_at_full_sample(self, config: AppConfig) -> None:
        full = config.features.h2h.confidence_full_sample
        meetings = [_meeting(i, winner=1) for i in range(full)]  # all p1 wins
        out = _extractor(meetings, config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_p1_win_rate"] == 1.0
        assert out["h2h_win_rate_confidence"] == pytest.approx(1.0)

    def test_weighted_recent_dominates(self, config: AppConfig) -> None:
        meetings = [
            _meeting(1, winner=1, start_ts=_AS_OF - timedelta(days=30)),   # recent win
            _meeting(2, winner=2, start_ts=_AS_OF - timedelta(days=900)),  # old loss
        ]
        out = _extractor(meetings, config).extract(_fctx(_current(), _AS_OF))
        # Simple rate is 0.5; recency weighting pulls it above 0.5.
        assert out["h2h_p1_win_rate"] == 0.5
        assert out["h2h_win_rate_weighted"] > 0.5

    def test_weighted_equals_rate_when_same_age(self, config: AppConfig) -> None:
        inst = _AS_OF - timedelta(days=100)
        meetings = [
            _meeting(1, winner=1, start_ts=inst),
            _meeting(2, winner=2, start_ts=inst),
        ]
        out = _extractor(meetings, config).extract(_fctx(_current(), _AS_OF))
        assert out["h2h_win_rate_weighted"] == pytest.approx(0.5)
