"""Tests for the Surface affinity + transition family (R5a) — `features/surface.py`.

Exercises the pure helpers (`_counts_as_played`, `_rate`, `_transition_type`)
and the `SurfaceExtractor`: career vs recent-365d surface win-rates, the `< 3`
recent-window NULL, the affinity diff and its NULL propagation, the §M2
transition label (debut `"none"`, no-change `"same"`, cross `"clay->hard"`,
unresolved-previous NULL), the `log1p` exposure within the adaptation window,
C14 counting, surface resolution via the tournament repo, PIT exclusion at the
cut, and the §M7 lockstep round-trip against the seeded `feature_specs` rows.

All in-memory; no Docker.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.surface import (
    SURFACE_FAMILY,
    SURFACE_FEATURE_KEYS,
    SurfaceExtractor,
    _counts_as_played,
    _rate,
    _transition_type,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.storage.postgres.rows import MatchRow, Surface, TournamentRow

_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
_P1 = 10
_P2 = 20
_OPP = 99  # dummy opponent so prior matches index under the player under test

# tournament_id → surface, shared across tests.
_SURFACES: dict[int, Surface] = {900: "Hard", 901: "Clay", 902: "Grass"}


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


class _CountingTournamentRepo(_FakeTournamentRepo):
    """A `_FakeTournamentRepo` that counts `get` calls — used to assert the
    surface memo bounds repo round-trips (Codex R5a)."""

    def __init__(self, surfaces: dict[int, Surface]) -> None:
        super().__init__(surfaces)
        self.calls = 0

    def get(self, tournament_id: int) -> TournamentRow | None:
        self.calls += 1
        return super().get(tournament_id)


def _match(
    *,
    match_id: int,
    player_id: int,
    won: bool,
    days_ago: int,
    tournament_id: int = 900,
    retired: bool = False,
    walkover: bool = False,
) -> MatchRow:
    """A prior match for `player_id` (as p1 vs a dummy opponent), `days_ago`
    before `_AS_OF`, on `tournament_id`'s surface, won/lost by the player."""
    inst = _AS_OF - timedelta(days=days_ago)
    return MatchRow(
        match_id=match_id,
        tournament_id=tournament_id,
        round="R32",
        match_date=inst.date(),
        p1_id=player_id,
        p2_id=_OPP,
        status="final",
        source="sackmann",
        source_uid=f"uid-{match_id}",
        start_ts=inst,
        winner_id=player_id if won else _OPP,
        retired=retired,
        walkover=walkover,
    )


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


def _fctx(surface: Surface = "Hard") -> FeatureContext:
    return FeatureContext(
        match=_current(),
        as_of_ts=_AS_OF,
        feature_set="v1",
        surface=surface,
        indoor=False,
        venue_id=None,
        tier="ATP500",
    )


def _extractor(
    matches: list[MatchRow],
    config: AppConfig,
    surfaces: dict[int, Surface] | None = None,
) -> SurfaceExtractor:
    return SurfaceExtractor(
        history=MatchHistoryIndex.build(matches),
        tournament_repo=_FakeTournamentRepo(
            surfaces if surfaces is not None else _SURFACES
        ),
        config=config,
    )


# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_rate_null_for_zero_total(self) -> None:
        assert _rate(0, 0) is None

    def test_rate_value(self) -> None:
        assert _rate(2, 3) == pytest.approx(2 / 3)

    def test_rate_min_total_guard(self) -> None:
        # The recent rate is NULL below the min_total threshold.
        assert _rate(2, 2, min_total=3) is None
        assert _rate(3, 3, min_total=3) == 1.0

    def test_transition_same_when_unchanged(self) -> None:
        assert _transition_type("Hard", "Hard") == "same"

    def test_transition_lowercased_pair(self) -> None:
        assert _transition_type("Clay", "Hard") == "clay->hard"

    def test_counts_walkover_excluded_retirement_kept(self) -> None:
        ret = _match(match_id=1, player_id=_P1, won=True, days_ago=10, retired=True)
        wo = _match(match_id=2, player_id=_P1, won=True, days_ago=10, walkover=True)
        assert _counts_as_played(ret, retirement_counts=True, walkover_counts=False)
        assert not _counts_as_played(wo, retirement_counts=True, walkover_counts=False)


class TestSurfaceContract:
    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ex = _extractor([], config)
        assert isinstance(ex, FeatureExtractor)
        assert ex.name == SURFACE_FAMILY

    def test_feature_keys_equal_seeded_rows(self, config: AppConfig) -> None:
        ex = _extractor([], config)
        seeded = {row.feature_key for row in _REGISTRY["surface"]}
        assert set(ex.feature_keys()) == seeded
        assert set(SURFACE_FEATURE_KEYS) == seeded

    def test_debut_all_rates_null_exposure_zero(self, config: AppConfig) -> None:
        out = _extractor([], config).extract(_fctx())
        assert out["p1_career_win_rate_surface"] is None
        assert out["p2_career_win_rate_surface"] is None
        assert out["p1_recent_win_rate_surface_365d"] is None
        assert out["surface_affinity_diff"] is None
        assert out["surface_transition_type"] == "none"
        assert out["surface_transition_exposure"] == 0.0

    def test_dtypes_native_python(self, config: AppConfig) -> None:
        matches = [
            _match(match_id=i, player_id=_P1, won=(i % 2 == 0), days_ago=10 + i)
            for i in range(3)
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert isinstance(out["p1_career_win_rate_surface"], float)
        assert isinstance(out["p1_recent_win_rate_surface_365d"], float)
        assert isinstance(out["surface_transition_type"], str)
        assert isinstance(out["surface_transition_exposure"], float)


class TestSurfaceWinRates:
    def test_career_win_rate(self, config: AppConfig) -> None:
        # 2 Hard wins, 1 Hard loss → career 2/3.
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=True, days_ago=20),
            _match(match_id=3, player_id=_P1, won=False, days_ago=30),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["p1_career_win_rate_surface"] == pytest.approx(2 / 3)
        assert out["p1_recent_win_rate_surface_365d"] == pytest.approx(2 / 3)

    def test_recent_null_below_three_matches(self, config: AppConfig) -> None:
        # 2 surface matches → career defined, recent NULL (< 3).
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=False, days_ago=20),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["p1_career_win_rate_surface"] == 0.5
        assert out["p1_recent_win_rate_surface_365d"] is None

    def test_only_matching_surface_counted(self, config: AppConfig) -> None:
        # 3 Hard wins + a Clay loss; the Clay match must not lower the Hard rate.
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=900),
            _match(match_id=2, player_id=_P1, won=True, days_ago=20, tournament_id=900),
            _match(match_id=3, player_id=_P1, won=True, days_ago=30, tournament_id=900),
            _match(match_id=4, player_id=_P1, won=False, days_ago=40, tournament_id=901),
        ]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["p1_career_win_rate_surface"] == 1.0

    def test_recent_window_excludes_old_matches(self, config: AppConfig) -> None:
        # 3 recent + 2 older-than-365d; recent rate uses only the in-window 3.
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=True, days_ago=20),
            _match(match_id=3, player_id=_P1, won=False, days_ago=30),
            _match(match_id=4, player_id=_P1, won=False, days_ago=400),
            _match(match_id=5, player_id=_P1, won=False, days_ago=500),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["p1_recent_win_rate_surface_365d"] == pytest.approx(2 / 3)
        assert out["p1_career_win_rate_surface"] == pytest.approx(2 / 5)

    def test_affinity_diff_is_p1_minus_p2(self, config: AppConfig) -> None:
        matches = [
            # p1: 3 Hard wins → recent 1.0
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=True, days_ago=20),
            _match(match_id=3, player_id=_P1, won=True, days_ago=30),
            # p2: 3 Hard, 1 win → recent 1/3
            _match(match_id=4, player_id=_P2, won=True, days_ago=10),
            _match(match_id=5, player_id=_P2, won=False, days_ago=20),
            _match(match_id=6, player_id=_P2, won=False, days_ago=30),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["surface_affinity_diff"] == pytest.approx(1.0 - 1 / 3)

    def test_affinity_diff_null_when_one_side_sparse(self, config: AppConfig) -> None:
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=True, days_ago=20),
            _match(match_id=3, player_id=_P1, won=True, days_ago=30),
            # p2 has only 1 surface match → recent NULL → diff NULL.
            _match(match_id=4, player_id=_P2, won=True, days_ago=10),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["p2_recent_win_rate_surface_365d"] is None
        assert out["surface_affinity_diff"] is None

    def test_unresolved_tournament_dropped_from_surface(self, config: AppConfig) -> None:
        # tournament 555 is not in the repo → its match is excluded from the
        # surface subset (no raise), leaving an empty surface history.
        matches = [_match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=555)]
        out = _extractor(matches, config).extract(_fctx())
        assert out["p1_career_win_rate_surface"] is None

    def test_match_at_cut_excluded(self, config: AppConfig) -> None:
        at_cut = _match(match_id=1, player_id=_P1, won=True, days_ago=0)  # instant == _AS_OF
        out = _extractor([at_cut], config).extract(_fctx())
        assert out["p1_career_win_rate_surface"] is None


class TestSurfaceTransition:
    def test_same_surface(self, config: AppConfig) -> None:
        matches = [_match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=900)]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["surface_transition_type"] == "same"

    def test_cross_surface_lowercased(self, config: AppConfig) -> None:
        # Previous match on Clay, current Hard.
        matches = [_match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=901)]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["surface_transition_type"] == "clay->hard"

    def test_uses_most_recent_prior_match(self, config: AppConfig) -> None:
        # Older Clay, more recent Grass → transition from Grass.
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=30, tournament_id=901),
            _match(match_id=2, player_id=_P1, won=True, days_ago=5, tournament_id=902),
        ]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["surface_transition_type"] == "grass->hard"

    def test_walkover_not_treated_as_previous_match(self, config: AppConfig) -> None:
        # Most recent is a walkover (excluded by C14); the last *played* match is
        # the Clay one → transition uses Clay, not the walkover's Grass.
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=20, tournament_id=901),
            _match(
                match_id=2, player_id=_P1, won=True, days_ago=5,
                tournament_id=902, walkover=True,
            ),
        ]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["surface_transition_type"] == "clay->hard"

    def test_null_when_previous_surface_unresolved(self, config: AppConfig) -> None:
        matches = [_match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=555)]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["surface_transition_type"] is None


class TestSurfaceCaching:
    def test_tournament_surface_resolved_once_per_distinct_id(
        self, config: AppConfig
    ) -> None:
        # 3 prior matches across 2 distinct tournaments, exercised over all three
        # surface passes (win-rate/transition/exposure) → the repo is hit exactly
        # once per distinct tournament_id, never once per match or per pass.
        repo = _CountingTournamentRepo(_SURFACES)
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=900),
            _match(match_id=2, player_id=_P1, won=True, days_ago=20, tournament_id=900),
            _match(match_id=3, player_id=_P1, won=False, days_ago=30, tournament_id=901),
        ]
        ex = SurfaceExtractor(
            history=MatchHistoryIndex.build(matches),
            tournament_repo=repo,
            config=config,
        )
        ex.extract(_fctx(surface="Hard"))
        assert repo.calls == 2


class TestSurfaceExposure:
    def test_log1p_count_within_window(self, config: AppConfig) -> None:
        # 2 Hard matches within 90d → log1p(2).
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=False, days_ago=80),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["surface_transition_exposure"] == pytest.approx(math.log1p(2))

    def test_excludes_matches_outside_adaptation_window(self, config: AppConfig) -> None:
        # 1 inside 90d, 1 outside → log1p(1).
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10),
            _match(match_id=2, player_id=_P1, won=False, days_ago=120),
        ]
        out = _extractor(matches, config).extract(_fctx())
        assert out["surface_transition_exposure"] == pytest.approx(math.log1p(1))

    def test_counts_only_current_surface(self, config: AppConfig) -> None:
        # 1 Hard + 1 Clay within window, current Hard → log1p(1).
        matches = [
            _match(match_id=1, player_id=_P1, won=True, days_ago=10, tournament_id=900),
            _match(match_id=2, player_id=_P1, won=False, days_ago=20, tournament_id=901),
        ]
        out = _extractor(matches, config).extract(_fctx(surface="Hard"))
        assert out["surface_transition_exposure"] == pytest.approx(math.log1p(1))
