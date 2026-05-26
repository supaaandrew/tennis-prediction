"""Elo feature family (R3) — the first Research Agent feature extractor.

Two collaborating components share one set of pure helpers:

  - `EloWalk` — the chronological *builder*. Over `for_training` (`final`-only,
    C4) matches it reads each match's pre-match ratings, emits the §15.5 Elo
    fragment, then updates both ladders and appends to `elo_snapshots` (H7).
    It owns the in-memory career-match counter (§M9) that drives the variable
    K-factor (H10) and `reliability_low`.
  - `EloExtractor` — the `FeatureExtractor` Protocol impl for the prediction
    path. Reads pre-match ratings strictly before `fctx.as_of_ts`; the career
    counts that drive `reliability_low` are injected (the final counts from a
    historical `EloWalk`, per §M9).

PIT discipline (§8/§A14): every pre-match read is at `pit_cut(match)`, and each
post-match snapshot is stamped at the match's *terminal* instant — which is
strictly after that cut — so a read never sees the snapshot its own match
produced (self-exclusion). Walkovers do not move Elo (§M10).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, time
from typing import Any

from tennis.core.config import AppConfig, EloConfig
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import (
    EloSnapshotRepository,
    TournamentRepository,
)
from tennis.storage.postgres.rows import EloSnapshotRow, EloSurface, MatchRow
from tennis.agents.research.context import FeatureContext, match_instant
from tennis.agents.research.point_in_time import pit_cut

_logger = get_logger("tennis.agents.research.features.elo")

# The family name (metrics key in R6) and the surface-agnostic "overall" ladder.
ELO_FAMILY = "elo"
_OVERALL: EloSurface = "overall"

# The 9 keys this family emits (§15.5; confirmed 9, not the spec prose's "11").
# Must stay in lockstep with the `"elo"` rows seeded by `specs.py` — the
# round-trip test in `test_elo.py` guards the two against drift (§M7).
ELO_FEATURE_KEYS: tuple[str, ...] = (
    "p1_elo_pre",
    "p2_elo_pre",
    "p1_elo_surface_pre",
    "p2_elo_surface_pre",
    "p1_elo_blended_pre",
    "p2_elo_blended_pre",
    "elo_diff_blended",
    "p1_elo_reliability_low",
    "p2_elo_reliability_low",
)


# ---------------------------------------------------------------------------
# Pure helpers (no IO) — shared by the walk and the extractor.
# ---------------------------------------------------------------------------
def _expected_score(self_elo: float, opp_elo: float) -> float:
    """Logistic expected score `E = 1 / (1 + 10^((opp - self)/400))`."""
    return 1.0 / (1.0 + 10.0 ** ((opp_elo - self_elo) / 400.0))


def _k_factor(prior_match_count: int, elo_cfg: EloConfig) -> int:
    """Variable K (H10): `k_new_player` for a player's first
    `k_threshold_matches` career matches, then `k_established`.

    `prior_match_count` is the player's career total *before* this match, so
    counts 0..(threshold-1) get the new-player K and count==threshold (the
    player's (threshold+1)th match) is the first to get the established K.
    """
    if prior_match_count < elo_cfg.k_threshold_matches:
        return elo_cfg.k_new_player
    return elo_cfg.k_established


def _blend(elo_overall: float, elo_surface: float, elo_cfg: EloConfig) -> float:
    """`(1 - surface_blend)*overall + surface_blend*surface` (§15.5)."""
    b = elo_cfg.surface_blend
    return (1.0 - b) * elo_overall + b * elo_surface


def _terminal_instant(match: MatchRow) -> datetime:
    """The instant the match's Elo update becomes 'known' — its snapshot stamp.

    `start_ts` when set (live), else `match_date` end-of-day UTC (historical).
    This is intentionally conservative: combined with the historical
    `match_date − 1 day` cut and `get_latest_before`'s `<=`, a match only sees
    prior matches that completed on an earlier day. Critically it is always
    strictly after `pit_cut(match)`, so a pre-match read at the cut never
    returns the snapshot this same match produces (PIT self-exclusion).
    """
    if match.start_ts is not None:
        return match.start_ts
    return datetime.combine(match.match_date, time.max, tzinfo=UTC)


def _read_rating(
    elo_repo: EloSnapshotRepository,
    *,
    player_id: int,
    surface: EloSurface,
    as_of_ts: datetime,
    initial_rating: int,
) -> float:
    """Latest pre-cut rating for `(player, surface)`, or the 1500 cold-start
    fallback (H10) when the player has no snapshot. Never raises."""
    snap = elo_repo.get_latest_before(
        player_id=player_id, surface=surface, as_of_ts=as_of_ts
    )
    return float(initial_rating) if snap is None else snap.elo_rating


def _fragment(
    *,
    p1_overall: float,
    p2_overall: float,
    p1_surface: float,
    p2_surface: float,
    p1_count: int,
    p2_count: int,
    elo_cfg: EloConfig,
) -> dict[str, Any]:
    """Build the 9-key p1-perspective Elo fragment from pre-match state."""
    p1_blended = _blend(p1_overall, p1_surface, elo_cfg)
    p2_blended = _blend(p2_overall, p2_surface, elo_cfg)
    low = elo_cfg.min_reliable_matches
    return {
        "p1_elo_pre": float(p1_overall),
        "p2_elo_pre": float(p2_overall),
        "p1_elo_surface_pre": float(p1_surface),
        "p2_elo_surface_pre": float(p2_surface),
        "p1_elo_blended_pre": p1_blended,
        "p2_elo_blended_pre": p2_blended,
        "elo_diff_blended": p1_blended - p2_blended,
        "p1_elo_reliability_low": p1_count < low,
        "p2_elo_reliability_low": p2_count < low,
    }


# ---------------------------------------------------------------------------
# The chronological builder.
# ---------------------------------------------------------------------------
class EloWalk:
    """Chronological Elo builder over `final` matches (C4).

    Holds the in-memory career-match counter (§M9) and appends to the
    append-only `elo_snapshots` ladder (H7). A single pass; out-of-order input
    is sorted into the §M6 chronological order first. `run` returns the
    per-match feature fragments (training-side emission); `career_counts`
    exposes the final counts to seed the prediction-path `EloExtractor`.
    """

    def __init__(
        self,
        *,
        elo_repo: EloSnapshotRepository,
        tournament_repo: TournamentRepository,
        config: AppConfig,
    ) -> None:
        self._elo_repo = elo_repo
        self._tournament_repo = tournament_repo
        self._config = config
        self._elo_cfg = config.features.elo
        self._counts: dict[int, int] = {}
        self._skipped_invalid_winner = 0

    @property
    def career_counts(self) -> Mapping[int, int]:
        """A copy of the final per-player career counts after `run` (§M9)."""
        return dict(self._counts)

    @property
    def skipped_invalid_winner(self) -> int:
        """Count of `final` matches skipped because `winner_id` was missing or
        not one of the two players. R3 does not abort the batch (one bad row
        must not kill the historical walk) nor dead-letter (no such surface in
        this layer); it surfaces the count so R6 can apply a threshold-to-fail /
        escalation policy instead of the skip being silent."""
        return self._skipped_invalid_winner

    def run(self, matches: Iterable[MatchRow]) -> dict[int, dict[str, Any]]:
        """Walk `matches` in chronological order; write snapshots and return
        `{match_id: fragment}`.

        Single-shot (§M9): `elo_snapshots` is append-only, so a rebuild is a
        full replay against an empty/truncated table — rerunning over populated
        history raises `IdempotencyError` on the first duplicate insert, by
        design (the limitation is loud, not silent). The in-memory career
        counter is a byproduct, exposed via `career_counts`.
        """
        # Reject a naive start_ts up front (mirrors MatchHistoryIndex.build):
        # otherwise the aware-vs-naive comparison inside `sorted` below raises an
        # opaque TypeError mid-batch instead of a clear, match-scoped ValueError.
        materialized = list(matches)
        for m in materialized:
            if m.start_ts is not None and m.start_ts.tzinfo is None:
                raise ValueError(
                    f"EloWalk: match {m.match_id} has a naive start_ts; "
                    "all timestamps must be timezone-aware UTC"
                )
        ordered = sorted(materialized, key=lambda m: (match_instant(m), m.match_id))
        offset = self._config.decision_timing.live_decision_offset_hours
        initial = self._elo_cfg.initial_rating
        fragments: dict[int, dict[str, Any]] = {}

        for match in ordered:
            surface = self._resolve_surface(match)
            if surface is None:
                # No resolvable surface -> cannot emit the (critical) surface
                # ratings; skip entirely. Logged in `_resolve_surface`.
                continue
            as_of = pit_cut(match, live_offset_hours=offset)
            p1, p2 = match.p1_id, match.p2_id

            p1_overall = _read_rating(
                self._elo_repo, player_id=p1, surface=_OVERALL,
                as_of_ts=as_of, initial_rating=initial,
            )
            p2_overall = _read_rating(
                self._elo_repo, player_id=p2, surface=_OVERALL,
                as_of_ts=as_of, initial_rating=initial,
            )
            p1_surface = _read_rating(
                self._elo_repo, player_id=p1, surface=surface,
                as_of_ts=as_of, initial_rating=initial,
            )
            p2_surface = _read_rating(
                self._elo_repo, player_id=p2, surface=surface,
                as_of_ts=as_of, initial_rating=initial,
            )
            p1_count = self._counts.get(p1, 0)
            p2_count = self._counts.get(p2, 0)

            fragments[match.match_id] = _fragment(
                p1_overall=p1_overall, p2_overall=p2_overall,
                p1_surface=p1_surface, p2_surface=p2_surface,
                p1_count=p1_count, p2_count=p2_count, elo_cfg=self._elo_cfg,
            )

            self._update(
                match, surface,
                p1_overall=p1_overall, p2_overall=p2_overall,
                p1_surface=p1_surface, p2_surface=p2_surface,
                p1_count=p1_count, p2_count=p2_count,
            )
        return fragments

    # ------------------------------------------------------------------
    def _resolve_surface(self, match: MatchRow) -> EloSurface | None:
        tournament = self._tournament_repo.get(match.tournament_id)
        if tournament is None:
            _logger.warning(
                "elo_surface_unresolved",
                match_id=match.match_id,
                tournament_id=match.tournament_id,
            )
            return None
        return tournament.surface  # Surface ⊂ EloSurface

    def _update(
        self,
        match: MatchRow,
        surface: EloSurface,
        *,
        p1_overall: float,
        p2_overall: float,
        p1_surface: float,
        p2_surface: float,
        p1_count: int,
        p2_count: int,
    ) -> None:
        """Apply the post-match Elo update to both ladders and append 4
        snapshots, then advance the career counter — UNLESS this is a walkover
        (§M10: no contest → no change, no snapshot, not counted). Retirements
        update normally (the result stands)."""
        if match.walkover:
            _logger.debug("elo_walkover_skipped", match_id=match.match_id)
            return
        p1, p2 = match.p1_id, match.p2_id
        winner = match.winner_id
        if winner not in (p1, p2):
            # `final` rows should always carry a valid winner; guard dirty data
            # rather than silently scoring it as a p2 win. Counted (not silent)
            # and surfaced via `skipped_invalid_winner` for R6 to escalate; not
            # raised, so one bad row can't kill the whole historical walk.
            self._skipped_invalid_winner += 1
            _logger.warning(
                "elo_update_skipped_no_winner",
                match_id=match.match_id,
                winner_id=winner,
            )
            return

        s1 = 1.0 if winner == p1 else 0.0
        s2 = 1.0 - s1
        k1 = _k_factor(p1_count, self._elo_cfg)  # per-player K (H10)
        k2 = _k_factor(p2_count, self._elo_cfg)
        stamp = _terminal_instant(match)

        # Overall ladder.
        self._write(p1, _OVERALL, p1_overall, p2_overall, s1, k1, stamp, match.match_id)
        self._write(p2, _OVERALL, p2_overall, p1_overall, s2, k2, stamp, match.match_id)
        # Surface ladder (independent state, same outcome + per-player K).
        self._write(p1, surface, p1_surface, p2_surface, s1, k1, stamp, match.match_id)
        self._write(p2, surface, p2_surface, p1_surface, s2, k2, stamp, match.match_id)

        # §M9: surface-agnostic career total, once per player per counted match.
        self._counts[p1] = p1_count + 1
        self._counts[p2] = p2_count + 1

    def _write(
        self,
        player_id: int,
        surface: EloSurface,
        self_elo: float,
        opp_elo: float,
        score: float,
        k: int,
        stamp: datetime,
        match_id: int,
    ) -> None:
        expected = _expected_score(self_elo, opp_elo)
        new_rating = self_elo + k * (score - expected)
        self._elo_repo.insert(
            EloSnapshotRow(
                player_id=player_id,
                surface=surface,
                elo_rating=new_rating,
                as_of_ts=stamp,
                match_id=match_id,
            )
        )


# ---------------------------------------------------------------------------
# The Protocol extractor (prediction path).
# ---------------------------------------------------------------------------
class EloExtractor:
    """`FeatureExtractor` for the Elo family.

    Reads pre-match ratings from `elo_snapshots` strictly before
    `fctx.as_of_ts` and returns the §15.5 fragment. The career counts that
    drive `reliability_low` (§M9) are a **required** constructor argument — for
    live prediction these are the final counts from a historical `EloWalk`;
    pass `{}` only for a genuinely cold system. Stateless given its injected
    dependencies (matches the `FeatureExtractor` contract).
    """

    name = ELO_FAMILY

    def __init__(
        self,
        *,
        elo_repo: EloSnapshotRepository,
        config: AppConfig,
        career_counts: Mapping[int, int],
    ) -> None:
        self._elo_repo = elo_repo
        self._elo_cfg = config.features.elo
        # Required, not optional: an implicit empty default would silently mark
        # EVERY player reliability_low (count 0 < min_reliable) if a caller
        # forgot to inject the walk's final counts. Callers pass an explicit
        # `{}` for a genuinely cold system with no history.
        self._career_counts: Mapping[int, int] = dict(career_counts)

    def feature_keys(self) -> tuple[str, ...]:
        return ELO_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        match = fctx.match
        as_of = fctx.as_of_ts
        surface = fctx.surface  # EloSurface-compatible (Surface ⊂ EloSurface)
        p1, p2 = match.p1_id, match.p2_id
        initial = self._elo_cfg.initial_rating
        return _fragment(
            p1_overall=_read_rating(
                self._elo_repo, player_id=p1, surface=_OVERALL,
                as_of_ts=as_of, initial_rating=initial,
            ),
            p2_overall=_read_rating(
                self._elo_repo, player_id=p2, surface=_OVERALL,
                as_of_ts=as_of, initial_rating=initial,
            ),
            p1_surface=_read_rating(
                self._elo_repo, player_id=p1, surface=surface,
                as_of_ts=as_of, initial_rating=initial,
            ),
            p2_surface=_read_rating(
                self._elo_repo, player_id=p2, surface=surface,
                as_of_ts=as_of, initial_rating=initial,
            ),
            p1_count=self._career_counts.get(p1, 0),
            p2_count=self._career_counts.get(p2, 0),
            elo_cfg=self._elo_cfg,
        )
