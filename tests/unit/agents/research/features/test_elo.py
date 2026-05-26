"""Tests for the Elo feature family (R3) — `agents/research/features/elo.py`.

Exercises the pure helpers, the chronological `EloWalk` builder (snapshots,
cold start, variable K, retirement-vs-walkover §M10, ladder independence,
chronology + PIT self-exclusion, the in-memory career counter §M9), the
`EloExtractor` Protocol impl, and the `feature_specs` lockstep round-trip (§M7).

All fakes are in-memory; no Docker. `FrozenClock` is unnecessary — `pit_cut`
derives the cut from each match's own fields, so the walk has no clock.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest

from tennis.agents.research.context import FeatureContext
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.elo import (
    ELO_FAMILY,
    ELO_FEATURE_KEYS,
    EloExtractor,
    EloWalk,
    _blend,
    _expected_score,
    _fragment,
    _k_factor,
    _terminal_instant,
)
from tennis.agents.research.point_in_time import pit_cut
from tennis.agents.research.specs import (
    _REGISTRY,
    build_expected_specs,
    seed_feature_specs,
)
from tennis.agents.research.validator import _CRITICAL_FEATURE_KEYS, FeatureSpec
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import FeatureContractError, IdempotencyError
from tennis.storage.postgres.rows import (
    EloSnapshotRow,
    EloSurface,
    FeatureSpecRow,
    MatchRow,
    Surface,
    TournamentRow,
)

_LIVE_OFFSET_H = 24  # config.decision_timing.live_decision_offset_hours default


# ---------------------------------------------------------------------------
# Fixtures + fakes
# ---------------------------------------------------------------------------
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
    tournament_id: int = 900,
    retired: bool = False,
    walkover: bool = False,
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


class _FakeEloRepo:
    """In-memory append-only `EloSnapshotRepository`.

    `get_latest_before` mirrors the impl: rows for (player, surface) with
    `as_of_ts <= cut`, latest by `as_of_ts` (insertion order breaks ties).
    `insert` enforces the append-only PK (player_id, surface, match_id) — a
    duplicate raises `IdempotencyError`, exactly like `EloSnapshotRepositoryImpl`.
    """

    def __init__(self) -> None:
        self.rows: list[EloSnapshotRow] = []
        self._keys: set[tuple[int, EloSurface, int]] = set()

    def insert(self, row: EloSnapshotRow) -> EloSnapshotRow:
        key = (row.player_id, row.surface, row.match_id)
        if key in self._keys:
            raise IdempotencyError(f"elo_snapshot already exists for {key}")
        self._keys.add(key)
        self.rows.append(row)
        return row

    def get_latest_before(
        self, *, player_id: int, surface: EloSurface, as_of_ts: datetime
    ) -> EloSnapshotRow | None:
        candidates = [
            (i, r)
            for i, r in enumerate(self.rows)
            if r.player_id == player_id
            and r.surface == surface
            and r.as_of_ts <= as_of_ts
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda ir: (ir[1].as_of_ts, ir[0]))[1]

    def count_for(self, player_id: int, surface: EloSurface) -> int:
        return sum(
            1 for r in self.rows if r.player_id == player_id and r.surface == surface
        )


class _FakeTournamentRepo:
    """Returns a `TournamentRow` with the configured surface per id, or None
    for unknown ids when no default is set (exercises the skip path)."""

    def __init__(
        self,
        *,
        surface_by_id: dict[int, Surface] | None = None,
        default: Surface | None = "Hard",
    ) -> None:
        self._by_id = dict(surface_by_id or {})
        self._default = default

    def get(self, tournament_id: int) -> TournamentRow | None:
        surface = self._by_id.get(tournament_id, self._default)
        if surface is None:
            return None
        return TournamentRow(
            tournament_id=tournament_id,
            season=2020,
            slug=f"t{tournament_id}",
            name=f"Tournament {tournament_id}",
            tier="ATP500",
            surface=surface,
            indoor=False,
        )


def _walk(config: AppConfig, **repos: Any) -> EloWalk:
    return EloWalk(
        elo_repo=repos.get("elo_repo") or _FakeEloRepo(),
        tournament_repo=repos.get("tournament_repo") or _FakeTournamentRepo(),
        config=config,
    )


def _fctx(match: MatchRow, *, as_of: datetime, surface: Surface = "Hard") -> FeatureContext:
    return FeatureContext(
        match=match,
        as_of_ts=as_of,
        feature_set="v1",
        surface=surface,
        indoor=False,
        venue_id=None,
        tier="ATP500",
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_expected_score_equal_ratings_is_half(self) -> None:
        assert _expected_score(1500.0, 1500.0) == pytest.approx(0.5)

    def test_expected_score_known_value(self) -> None:
        # 1/(1+10^(200/400)) = 1/(1+sqrt(10)) ≈ 0.2403.
        assert _expected_score(1500.0, 1700.0) == pytest.approx(0.24025, abs=1e-4)

    def test_expected_scores_sum_to_one(self) -> None:
        a, b = 1623.0, 1481.0
        assert _expected_score(a, b) + _expected_score(b, a) == pytest.approx(1.0)

    def test_expected_score_favours_higher_rating(self) -> None:
        assert _expected_score(1800.0, 1500.0) > 0.5

    def test_k_factor_below_threshold_is_new_player(self, config: AppConfig) -> None:
        cfg = config.features.elo
        assert _k_factor(cfg.k_threshold_matches - 1, cfg) == cfg.k_new_player

    def test_k_factor_at_threshold_is_established(self, config: AppConfig) -> None:
        # The boundary: count == threshold is the first match to drop to the
        # established K (the player's (threshold+1)th match).
        cfg = config.features.elo
        assert _k_factor(cfg.k_threshold_matches, cfg) == cfg.k_established

    def test_k_factor_zero_is_new_player(self, config: AppConfig) -> None:
        cfg = config.features.elo
        assert _k_factor(0, cfg) == cfg.k_new_player

    def test_k_factor_large_is_established(self, config: AppConfig) -> None:
        cfg = config.features.elo
        assert _k_factor(10_000, cfg) == cfg.k_established

    def test_blend_formula(self, config: AppConfig) -> None:
        cfg = config.features.elo
        b = cfg.surface_blend
        assert _blend(1500.0, 1700.0, cfg) == pytest.approx(
            (1.0 - b) * 1500.0 + b * 1700.0
        )

    def test_blend_equal_inputs_is_identity(self, config: AppConfig) -> None:
        assert _blend(1600.0, 1600.0, config.features.elo) == pytest.approx(1600.0)

    def test_terminal_instant_historical_is_end_of_day(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 5, 1))
        stamp = _terminal_instant(m)
        assert stamp == datetime.combine(date(2020, 5, 1), time.max, tzinfo=UTC)
        assert stamp.tzinfo is not None

    def test_terminal_instant_live_is_start_ts(self) -> None:
        start = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
        m = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2026, 5, 1), start_ts=start
        )
        assert _terminal_instant(m) == start

    def test_terminal_instant_strictly_after_pit_cut(self) -> None:
        # The PIT self-exclusion invariant: snapshot stamp > cut, both states.
        hist = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 5, 1))
        live = _match(
            match_id=2,
            p1_id=1,
            p2_id=2,
            match_date=date(2026, 5, 1),
            start_ts=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
        )
        for m in (hist, live):
            assert _terminal_instant(m) > pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)


class TestFragment:
    def test_has_exactly_the_nine_keys(self, config: AppConfig) -> None:
        frag = _fragment(
            p1_overall=1500.0, p2_overall=1500.0,
            p1_surface=1500.0, p2_surface=1500.0,
            p1_count=0, p2_count=0, elo_cfg=config.features.elo,
        )
        assert set(frag) == set(ELO_FEATURE_KEYS)
        assert len(frag) == 9

    def test_diff_sign_is_p1_minus_p2(self, config: AppConfig) -> None:
        frag = _fragment(
            p1_overall=1700.0, p2_overall=1500.0,
            p1_surface=1700.0, p2_surface=1500.0,
            p1_count=50, p2_count=50, elo_cfg=config.features.elo,
        )
        assert frag["elo_diff_blended"] > 0
        assert frag["elo_diff_blended"] == pytest.approx(
            frag["p1_elo_blended_pre"] - frag["p2_elo_blended_pre"]
        )

    def test_ratings_are_floats_reliability_is_bool(self, config: AppConfig) -> None:
        frag = _fragment(
            p1_overall=1500.0, p2_overall=1500.0,
            p1_surface=1500.0, p2_surface=1500.0,
            p1_count=0, p2_count=99, elo_cfg=config.features.elo,
        )
        assert isinstance(frag["p1_elo_pre"], float)
        assert isinstance(frag["elo_diff_blended"], float)
        assert isinstance(frag["p1_elo_reliability_low"], bool)

    def test_reliability_low_true_below_threshold(self, config: AppConfig) -> None:
        cfg = config.features.elo
        frag = _fragment(
            p1_overall=1500.0, p2_overall=1500.0,
            p1_surface=1500.0, p2_surface=1500.0,
            p1_count=cfg.min_reliable_matches - 1,
            p2_count=cfg.min_reliable_matches,
            elo_cfg=cfg,
        )
        assert frag["p1_elo_reliability_low"] is True
        assert frag["p2_elo_reliability_low"] is False  # boundary: == is reliable


# ---------------------------------------------------------------------------
# EloWalk — cold start + snapshot writing
# ---------------------------------------------------------------------------
class TestEloWalkColdStart:
    def test_first_match_pre_ratings_are_cold_start(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)

        frags = walk.run([m])

        init = float(config.features.elo.initial_rating)
        frag = frags[1]
        assert frag["p1_elo_pre"] == init
        assert frag["p2_elo_pre"] == init
        assert frag["p1_elo_surface_pre"] == init
        assert frag["elo_diff_blended"] == pytest.approx(0.0)

    def test_first_match_updates_with_new_player_k(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)

        walk.run([m])

        cfg = config.features.elo
        init = float(cfg.initial_rating)
        expected_winner = init + cfg.k_new_player * (1.0 - 0.5)
        expected_loser = init + cfg.k_new_player * (0.0 - 0.5)
        winner_snap = elo.get_latest_before(
            player_id=1, surface="overall", as_of_ts=_terminal_instant(m)
        )
        loser_snap = elo.get_latest_before(
            player_id=2, surface="overall", as_of_ts=_terminal_instant(m)
        )
        assert winner_snap is not None and winner_snap.elo_rating == pytest.approx(expected_winner)
        assert loser_snap is not None and loser_snap.elo_rating == pytest.approx(expected_loser)

    def test_writes_four_snapshots_per_counted_match(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)

        walk.run([m])

        assert len(elo.rows) == 4
        # one per (player, ladder)
        assert elo.count_for(1, "overall") == 1
        assert elo.count_for(2, "overall") == 1
        assert elo.count_for(1, "Hard") == 1
        assert elo.count_for(2, "Hard") == 1

    def test_winner_is_p2_moves_p2_up(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=2)

        walk.run([m])

        p2 = elo.get_latest_before(player_id=2, surface="overall", as_of_ts=_terminal_instant(m))
        assert p2 is not None and p2.elo_rating > config.features.elo.initial_rating

    def test_empty_input_writes_nothing(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)

        assert walk.run([]) == {}
        assert elo.rows == []

    def test_unresolved_tournament_skips_match(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        # default=None -> get() returns None for the unknown tournament id.
        walk = EloWalk(
            elo_repo=elo,
            tournament_repo=_FakeTournamentRepo(default=None),
            config=config,
        )
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)

        frags = walk.run([m])

        assert frags == {}
        assert elo.rows == []


# ---------------------------------------------------------------------------
# EloWalk — chronology + PIT
# ---------------------------------------------------------------------------
class TestEloWalkChronologyAndPit:
    def test_out_of_order_input_processed_in_date_order(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        early = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        # Far enough later (>1 day) to clear the conservative cut and chain.
        late = _match(match_id=2, p1_id=1, p2_id=3, match_date=date(2020, 6, 1), winner_id=1)

        frags = walk.run([late, early])  # deliberately out of order

        init = float(config.features.elo.initial_rating)
        assert frags[1]["p1_elo_pre"] == init  # early match: cold start
        assert frags[2]["p1_elo_pre"] > init   # late match: reflects the win

    def test_pit_self_exclusion(self, config: AppConfig) -> None:
        # The match's own post-match snapshot (stamped at the terminal instant)
        # must be invisible to a read at that match's pit_cut.
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)

        walk.run([m])

        cut = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)
        assert elo.get_latest_before(player_id=1, surface="overall", as_of_ts=cut) is None

    def test_adjacent_day_does_not_chain(self, config: AppConfig) -> None:
        # Conservative PIT: end-of-day stamp + (match_date-1) cut means a match
        # cannot see a result from the immediately preceding day.
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        day1 = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        day2 = _match(match_id=2, p1_id=1, p2_id=3, match_date=date(2020, 1, 2), winner_id=1)

        frags = walk.run([day1, day2])

        assert frags[2]["p1_elo_pre"] == float(config.features.elo.initial_rating)

    def test_two_day_gap_chains(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        d1 = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        d3 = _match(match_id=2, p1_id=1, p2_id=3, match_date=date(2020, 1, 3), winner_id=1)

        frags = walk.run([d1, d3])

        assert frags[2]["p1_elo_pre"] > float(config.features.elo.initial_rating)

    def test_accumulates_across_two_updates(self, config: AppConfig) -> None:
        # Two wins for player 1 on the overall ladder compound exactly.
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m1 = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        m2 = _match(match_id=2, p1_id=1, p2_id=3, match_date=date(2020, 3, 1), winner_id=1)

        frags = walk.run([m1, m2])

        cfg = config.features.elo
        init = float(cfg.initial_rating)
        r1 = init + cfg.k_new_player * (1.0 - 0.5)  # after match 1
        e2 = _expected_score(r1, init)
        r2 = r1 + cfg.k_new_player * (1.0 - e2)
        final = elo.get_latest_before(
            player_id=1, surface="overall", as_of_ts=_terminal_instant(m2)
        )
        assert frags[2]["p1_elo_pre"] == pytest.approx(r1)
        assert final is not None and final.elo_rating == pytest.approx(r2)

    def test_live_match_uses_start_ts_offset(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        start = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
        m = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2026, 5, 10),
            start_ts=start, winner_id=1,
        )

        walk.run([m])

        snap = elo.get_latest_before(player_id=1, surface="overall", as_of_ts=start)
        assert snap is not None and snap.as_of_ts == start

    def test_rejects_naive_start_ts(self, config: AppConfig) -> None:
        # Fail fast with a clear, match-scoped error rather than an opaque
        # TypeError from sorting aware against naive instants mid-batch.
        walk = _walk(config)
        bad = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2026, 5, 1),
            start_ts=datetime(2026, 5, 1, 12, 0), winner_id=1,  # naive
        )

        with pytest.raises(ValueError, match="timezone-aware"):
            walk.run([bad])

    def test_naive_start_ts_rejected_even_when_mixed(self, config: AppConfig) -> None:
        # The dangerous case: one naive row among aware/historical rows is what
        # would crash `sorted`. Validation must catch it before the sort.
        walk = _walk(config)
        good = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        bad = _match(
            match_id=2, p1_id=3, p2_id=4, match_date=date(2026, 5, 1),
            start_ts=datetime(2026, 5, 1, 12, 0), winner_id=3,  # naive
        )

        with pytest.raises(ValueError, match="match 2"):
            walk.run([good, bad])


class TestEloWalkReplaySemantics:
    def test_rerun_against_populated_repo_raises(self, config: AppConfig) -> None:
        # §M9: the walk is single-shot. A resumed run replays against the
        # already-populated append-only ladder -> duplicate PK -> loud failure,
        # by design. Tolerating it would silently mask rating drift.
        elo = _FakeEloRepo()
        trepo = _FakeTournamentRepo()
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)

        EloWalk(elo_repo=elo, tournament_repo=trepo, config=config).run([m])  # build

        with pytest.raises(IdempotencyError):
            # Fresh walk (rebuilt counter), same populated repo.
            EloWalk(elo_repo=elo, tournament_repo=trepo, config=config).run([m])

    def test_full_rebuild_into_empty_repo_is_deterministic(self, config: AppConfig) -> None:
        # The supported path: replay into a fresh/truncated table reproduces the
        # exact same snapshots and counter.
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        trepo = _FakeTournamentRepo()

        elo_a = _FakeEloRepo()
        walk_a = EloWalk(elo_repo=elo_a, tournament_repo=trepo, config=config)
        frags_a = walk_a.run([m])

        elo_b = _FakeEloRepo()
        walk_b = EloWalk(elo_repo=elo_b, tournament_repo=trepo, config=config)
        frags_b = walk_b.run([m])

        assert frags_a == frags_b
        assert dict(walk_a.career_counts) == dict(walk_b.career_counts)
        assert [r.elo_rating for r in elo_a.rows] == [r.elo_rating for r in elo_b.rows]


# ---------------------------------------------------------------------------
# EloWalk — retirement vs walkover (§M10) + the career counter (§M9)
# ---------------------------------------------------------------------------
class TestEloWalkRetirementWalkover:
    def test_retirement_updates_elo_and_counts(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1),
            winner_id=1, retired=True,
        )

        frags = walk.run([m])

        assert len(elo.rows) == 4  # retirement updates normally
        assert dict(walk.career_counts) == {1: 1, 2: 1}
        assert frags[1] is not None

    def test_walkover_skips_update_but_emits_fragment(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1),
            winner_id=1, walkover=True,
        )

        frags = walk.run([m])

        assert elo.rows == []                 # no Elo change
        assert dict(walk.career_counts) == {} # not counted toward career total
        # fragment still emitted (pre-match ratings exist regardless)
        assert frags[1]["p1_elo_pre"] == float(config.features.elo.initial_rating)

    def test_walkover_does_not_count_toward_later_match(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        wo = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1),
            winner_id=1, walkover=True,
        )
        real = _match(match_id=2, p1_id=1, p2_id=3, match_date=date(2020, 6, 1), winner_id=1)

        walk.run([wo, real])

        # Only the real match counted; player 1 has exactly 1 career match.
        assert dict(walk.career_counts)[1] == 1
        # And the real match saw cold-start ratings (walkover wrote no snapshot).
        snaps = [r for r in elo.rows if r.match_id == 1]
        assert snaps == []

    def test_missing_winner_skips_update_and_is_counted(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=None)

        frags = walk.run([m])

        assert elo.rows == []
        assert dict(walk.career_counts) == {}
        assert frags[1]["p1_elo_pre"] == float(config.features.elo.initial_rating)
        # Not silent: surfaced for R6 to escalate (threshold-to-fail).
        assert walk.skipped_invalid_winner == 1

    def test_winner_not_in_pair_is_counted(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        m = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=999
        )

        walk.run([m])

        assert elo.rows == []
        assert walk.skipped_invalid_winner == 1

    def test_skipped_counter_zero_on_clean_input(self, config: AppConfig) -> None:
        walk = _walk(config)
        walk.run(
            [_match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)]
        )
        assert walk.skipped_invalid_winner == 0

    def test_career_counts_returns_a_copy(self, config: AppConfig) -> None:
        walk = _walk(config)
        walk.run([_match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)])

        counts = walk.career_counts
        counts[1] = 999  # mutate the returned mapping
        assert dict(walk.career_counts)[1] == 1  # internal state untouched

    def test_first_match_reliability_low_for_both(self, config: AppConfig) -> None:
        walk = _walk(config)
        frags = walk.run(
            [_match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)]
        )
        assert frags[1]["p1_elo_reliability_low"] is True
        assert frags[1]["p2_elo_reliability_low"] is True


# ---------------------------------------------------------------------------
# EloWalk — ladder independence
# ---------------------------------------------------------------------------
class TestEloWalkLadderIndependence:
    def test_surface_ladder_independent_of_overall(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        trepo = _FakeTournamentRepo(surface_by_id={10: "Hard", 20: "Clay"})
        walk = EloWalk(elo_repo=elo, tournament_repo=trepo, config=config)
        # P1 wins on Hard, then plays on Clay months later.
        hard = _match(
            match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1),
            winner_id=1, tournament_id=10,
        )
        clay = _match(
            match_id=2, p1_id=1, p2_id=3, match_date=date(2020, 6, 1),
            winner_id=1, tournament_id=20,
        )

        frags = walk.run([hard, clay])

        init = float(config.features.elo.initial_rating)
        # Overall reflects the Hard win; Clay ladder is still cold-start.
        assert frags[2]["p1_elo_pre"] > init
        assert frags[2]["p1_elo_surface_pre"] == init

    def test_separate_snapshot_rows_per_surface(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        trepo = _FakeTournamentRepo(surface_by_id={10: "Clay"})
        walk = EloWalk(elo_repo=elo, tournament_repo=trepo, config=config)
        walk.run(
            [_match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1),
                    winner_id=1, tournament_id=10)]
        )

        assert elo.count_for(1, "Clay") == 1
        assert elo.count_for(1, "overall") == 1
        assert elo.count_for(1, "Hard") == 0


# ---------------------------------------------------------------------------
# EloExtractor — the Protocol path
# ---------------------------------------------------------------------------
class TestEloExtractor:
    def _ext(self, config: AppConfig, elo: _FakeEloRepo, **kw: Any) -> EloExtractor:
        kw.setdefault("career_counts", {})  # required arg; {} = cold system
        return EloExtractor(elo_repo=elo, config=config, **kw)

    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ext = self._ext(config, _FakeEloRepo())
        assert isinstance(ext, FeatureExtractor)

    def test_career_counts_is_required(self, config: AppConfig) -> None:
        # Omitting career_counts is a TypeError — no silent empty default that
        # would mark every player reliability_low in production (Codex MEDIUM).
        with pytest.raises(TypeError):
            EloExtractor(elo_repo=_FakeEloRepo(), config=config)  # type: ignore[call-arg]

    def test_name_and_feature_keys(self, config: AppConfig) -> None:
        ext = self._ext(config, _FakeEloRepo())
        assert ext.name == ELO_FAMILY == "elo"
        assert set(ext.feature_keys()) == set(ELO_FEATURE_KEYS)
        assert len(ext.feature_keys()) == 9

    def test_extract_cold_start(self, config: AppConfig) -> None:
        ext = self._ext(config, _FakeEloRepo())
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        as_of = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

        frag = ext.extract(_fctx(m, as_of=as_of))

        init = float(config.features.elo.initial_rating)
        assert frag["p1_elo_pre"] == init
        assert frag["p1_elo_reliability_low"] is True  # empty counts -> low

    def test_extract_reads_prior_snapshots(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        elo.insert(
            EloSnapshotRow(
                player_id=1, surface="overall", elo_rating=1650.0,
                as_of_ts=datetime(2019, 1, 1, tzinfo=UTC), match_id=99,
            )
        )
        ext = self._ext(config, elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        as_of = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

        frag = ext.extract(_fctx(m, as_of=as_of))

        assert frag["p1_elo_pre"] == pytest.approx(1650.0)

    def test_extract_uses_fctx_surface_ladder(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        elo.insert(
            EloSnapshotRow(
                player_id=1, surface="Clay", elo_rating=1720.0,
                as_of_ts=datetime(2019, 1, 1, tzinfo=UTC), match_id=99,
            )
        )
        ext = self._ext(config, elo)
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        as_of = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

        frag = ext.extract(_fctx(m, as_of=as_of, surface="Clay"))

        assert frag["p1_elo_surface_pre"] == pytest.approx(1720.0)

    def test_extract_reliability_from_injected_counts(self, config: AppConfig) -> None:
        cfg = config.features.elo
        ext = self._ext(
            config,
            _FakeEloRepo(),
            career_counts={1: cfg.min_reliable_matches, 2: cfg.min_reliable_matches - 1},
        )
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        as_of = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

        frag = ext.extract(_fctx(m, as_of=as_of))

        assert frag["p1_elo_reliability_low"] is False  # at threshold -> reliable
        assert frag["p2_elo_reliability_low"] is True

    def test_extract_does_not_see_snapshot_at_or_after_as_of(self, config: AppConfig) -> None:
        elo = _FakeEloRepo()
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        as_of = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)
        # A snapshot strictly after the cut must not leak into the read.
        elo.insert(
            EloSnapshotRow(
                player_id=1, surface="overall", elo_rating=1900.0,
                as_of_ts=as_of + timedelta(days=1), match_id=50,
            )
        )
        ext = self._ext(config, elo)

        frag = ext.extract(_fctx(m, as_of=as_of))

        assert frag["p1_elo_pre"] == float(config.features.elo.initial_rating)

    def test_extract_returns_all_nine_keys(self, config: AppConfig) -> None:
        ext = self._ext(config, _FakeEloRepo())
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        as_of = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

        frag = ext.extract(_fctx(m, as_of=as_of))

        assert set(frag) == set(ELO_FEATURE_KEYS)

    def test_walk_then_extract_share_state_via_counts(self, config: AppConfig) -> None:
        # End-to-end: build history, then inject the final counts into the
        # extractor for a future prediction.
        elo = _FakeEloRepo()
        walk = _walk(config, elo_repo=elo)
        hist = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1), winner_id=1)
        walk.run([hist])

        ext = EloExtractor(elo_repo=elo, config=config, career_counts=walk.career_counts)
        future = _match(match_id=2, p1_id=1, p2_id=9, match_date=date(2020, 6, 1))
        as_of = pit_cut(future, live_offset_hours=_LIVE_OFFSET_H)

        frag = ext.extract(_fctx(future, as_of=as_of))

        assert frag["p1_elo_pre"] > float(config.features.elo.initial_rating)
        # one career match < min_reliable -> still low
        assert frag["p1_elo_reliability_low"] is True


# ---------------------------------------------------------------------------
# feature_specs lockstep (§M7) — registry + round-trip
# ---------------------------------------------------------------------------
class _FakeFeatureSpecRepo:
    def __init__(self) -> None:
        self._store: dict[tuple[str, int], FeatureSpecRow] = {}

    def list_active(self, *, feature_set: str) -> Sequence[FeatureSpecRow]:
        return tuple(self._store.values())

    def upsert(self, row: FeatureSpecRow) -> FeatureSpecRow:
        self._store[(row.feature_key, row.version)] = row
        return row


class TestSpecsLockstep:
    def test_feature_keys_match_registry(self, config: AppConfig) -> None:
        # The drift guard: the extractor's keys must equal the seeded family.
        ext = EloExtractor(elo_repo=_FakeEloRepo(), config=config, career_counts={})
        registry_keys = {row.feature_key for row in _REGISTRY["elo"]}
        assert set(ext.feature_keys()) == registry_keys

    def test_seed_then_build_no_drift(self) -> None:
        repo = _FakeFeatureSpecRepo()
        n = seed_feature_specs(repo, families=["elo"])  # uses production _REGISTRY
        assert n == 9

        specs = build_expected_specs(repo, feature_set="v1", families=["elo"])
        assert len(specs) == 9
        assert all(isinstance(s, FeatureSpec) for s in specs)

    def test_critical_stamping(self) -> None:
        repo = _FakeFeatureSpecRepo()
        seed_feature_specs(repo, families=["elo"])
        specs = build_expected_specs(repo, feature_set="v1", families=["elo"])
        by_key = {s.feature_key: s for s in specs}

        # 7 base ratings critical (§M8); 2 reliability booleans not.
        assert by_key["elo_diff_blended"].critical is True
        assert by_key["p1_elo_pre"].critical is True
        assert by_key["p1_elo_reliability_low"].critical is False
        assert by_key["p2_elo_reliability_low"].critical is False

    def test_reliability_dtype_is_bool(self) -> None:
        repo = _FakeFeatureSpecRepo()
        seed_feature_specs(repo, families=["elo"])
        specs = build_expected_specs(repo, feature_set="v1", families=["elo"])
        by_key = {s.feature_key: s for s in specs}

        assert by_key["p1_elo_reliability_low"].dtype == "bool"
        assert by_key["p1_elo_pre"].dtype == "float"

    def test_build_without_seed_raises_drift(self) -> None:
        # Registered family requested but nothing seeded -> hard fail (§M7).
        repo = _FakeFeatureSpecRepo()
        with pytest.raises(FeatureContractError, match="catalog drift"):
            build_expected_specs(repo, feature_set="v1", families=["elo"])

    def test_critical_keyset_excludes_reliability(self) -> None:
        assert "p1_elo_reliability_low" not in _CRITICAL_FEATURE_KEYS
        assert "p1_elo_pre" in _CRITICAL_FEATURE_KEYS
