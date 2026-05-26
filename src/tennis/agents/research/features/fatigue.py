"""Fatigue feature family (R7) — `features.fatigue` (§15.5).

`FatigueExtractor` implements the `FeatureExtractor` Protocol for the `fatigue`
family. For each side it reads the player's prior matches strictly before
`fctx.as_of_ts` (§M6 PIT, via `MatchHistoryIndex`) and derives recent-load signals:

  - `p{1,2}_rest_days` — `(as_of.date() - last_played.match_date).days`, where
    `last_played` is the most recent prior match the player actually contested
    (retirements count; walkovers excluded — C14).
  - `p{1,2}_matches_last_{7,14}d` — weighted count of prior matches whose instant
    falls in `[as_of - Nd, as_of)`; retirement contributes
    `retirement_fatigue_weight`, walkover contributes 0 (both config-driven, C14).
  - `p{1,2}_minutes_last_{7,14}d` — weighted Σ `minutes` over the same window
    (full = 1.0×, retirement = `retirement_fatigue_weight`×). Pre-1991 rows carry
    no `minutes`; a counting window match with `minutes IS NULL` makes the sum NULL
    (can't form a complete load), mirroring serve_return's NULL-by-absence.
  - `p{1,2}_travel_km_since_last_match` — great-circle (haversine) distance between
    `last_played`'s venue and the current match venue. NULL when either venue or any
    coordinate is missing.

§M19-adjacent / catalog faithfulness: there is **no bo5 sets-equivalent weighting**
(the §15.5 catalog defines none, and `config.feature_engineering` has no such knob).
`best_of` is NOT read here — match load is weighted only by the C14 retirement/walkover
rules. The H8 `best_of_null_fallback_by_tier` is not consumed by any fatigue formula.

Every key is non-critical (§0.5/§M8): a debut player (no prior matches) yields all
keys NULL, the pre-OWM-geocoding era yields `travel_km` NULL, and pre-1991 minutes
absence yields the minutes keys NULL — none is ever a hard failure. A `StorageError`
while resolving venue coordinates degrades `travel_km` to NULL (it does not abort the
match); a genuine programming defect propagates to the agent's per-match dead-letter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.errors import StorageError
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import TournamentRepository, VenueRepository
from tennis.storage.postgres.rows import MatchRow
from tennis.agents.research.context import (
    FeatureContext,
    MatchHistoryIndex,
    match_instant,
)

_logger = get_logger("tennis.agents.research.features.fatigue")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
FATIGUE_FAMILY = "fatigue"

# The catalog-pinned trailing windows (days) for the recent-load counts (§15.5).
# These are fixed by the catalog, NOT `config.features.windows_days` (those are the
# Form windows). The KEYS tuple below is built from this — keep them in lockstep.
_FATIGUE_WINDOWS: tuple[int, ...] = (7, 14)

# Earth mean radius (km) for the haversine travel distance.
_EARTH_RADIUS_KM = 6371.0088

# Cache-miss sentinel — distinguishes "not yet looked up" from a cached None
# resolution (a venue/tournament that genuinely has no coordinates).
_UNSET: object = object()

# The 12 keys this family emits (§15.5 `features.fatigue`, in catalog order). Must
# stay in lockstep with the `"fatigue"` rows seeded by specs.py — guarded by the
# round-trip test (§M7).
FATIGUE_FEATURE_KEYS: tuple[str, ...] = (
    "p1_rest_days",
    "p2_rest_days",
    "p1_matches_last_7d",
    "p2_matches_last_7d",
    "p1_matches_last_14d",
    "p2_matches_last_14d",
    "p1_minutes_last_7d",
    "p2_minutes_last_7d",
    "p1_minutes_last_14d",
    "p2_minutes_last_14d",
    "p1_travel_km_since_last_match",
    "p2_travel_km_since_last_match",
)


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance (km) between two lat/lon points (degrees)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class FatigueExtractor:
    """`FeatureExtractor` for the `fatigue` family. Stateless given its injected
    `MatchHistoryIndex`, `TournamentRepository`, `VenueRepository`, and config.

    `tournament_repo` is needed because a *prior* match carries only its
    `tournament_id`; the travel feature resolves that to a venue (and its
    coordinates) — the current match's venue arrives ready on `fctx.venue_id`."""

    name = FATIGUE_FAMILY

    def __init__(
        self,
        *,
        history: MatchHistoryIndex,
        tournament_repo: TournamentRepository,
        venue_repo: VenueRepository,
        config: AppConfig,
    ) -> None:
        self._history = history
        self._tournament_repo = tournament_repo
        self._venue_repo = venue_repo
        fe = config.feature_engineering
        self._retirement_counts = fe.retirement_counts_as_match
        self._walkover_counts = fe.walkover_counts_as_match
        self._retirement_weight = fe.retirement_fatigue_weight
        # Run-scoped memoization (the extractor is built once per run): venue
        # coordinates and tournament→venue mappings are static within a run, so the
        # per-match/per-player travel resolution caches O(1) after first hit instead
        # of re-querying (retires the N+1 in the hot loop — mirrors §M18). Only clean
        # DATA resolutions are cached; a transient StorageError is NOT cached (so it
        # can't poison a venue into a permanent miss).
        self._venue_coords_cache: dict[int, tuple[float, float] | None] = {}
        self._tournament_venue_cache: dict[int, int | None] = {}

    def feature_keys(self) -> tuple[str, ...]:
        return FATIGUE_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        as_of = fctx.as_of_ts
        current_coords = self._coords_for_venue(fctx.venue_id)
        p1 = self._player_fatigue(
            fctx.match.p1_id, as_of=as_of, current_coords=current_coords
        )
        p2 = self._player_fatigue(
            fctx.match.p2_id, as_of=as_of, current_coords=current_coords
        )
        return {
            "p1_rest_days": p1["rest_days"],
            "p2_rest_days": p2["rest_days"],
            "p1_matches_last_7d": p1["matches_last_7d"],
            "p2_matches_last_7d": p2["matches_last_7d"],
            "p1_matches_last_14d": p1["matches_last_14d"],
            "p2_matches_last_14d": p2["matches_last_14d"],
            "p1_minutes_last_7d": p1["minutes_last_7d"],
            "p2_minutes_last_7d": p2["minutes_last_7d"],
            "p1_minutes_last_14d": p1["minutes_last_14d"],
            "p2_minutes_last_14d": p2["minutes_last_14d"],
            "p1_travel_km_since_last_match": p1["travel_km"],
            "p2_travel_km_since_last_match": p2["travel_km"],
        }

    # -- per-player aggregation --------------------------------------------
    def _player_fatigue(
        self,
        player_id: int,
        *,
        as_of: datetime,
        current_coords: tuple[float, float] | None,
    ) -> dict[str, Any]:
        """All six fatigue values for one player (unprefixed keys).

        A debut player (no prior matches) yields every key NULL — there is no
        fatigue signal to report (§M8). With any history, the window counts are a
        genuine 0.0 when the player simply rested; only debut, pre-1991 minutes
        absence, and unresolved venues drive NULL."""
        priors = self._history.player_matches_before(player_id=player_id, as_of=as_of)
        if not priors:
            return {
                "rest_days": None,
                "matches_last_7d": None,
                "matches_last_14d": None,
                "minutes_last_7d": None,
                "minutes_last_14d": None,
                "travel_km": None,
            }

        last_played = self._last_played(priors)
        rest_days = (
            (as_of.date() - last_played.match_date).days
            if last_played is not None
            else None
        )

        out: dict[str, Any] = {"rest_days": rest_days}
        for n in _FATIGUE_WINDOWS:
            lower = as_of - timedelta(days=n)
            window = [m for m in priors if match_instant(m) >= lower]
            out[f"matches_last_{n}d"] = self._matches_count(window)
            out[f"minutes_last_{n}d"] = self._minutes_sum(window)

        out["travel_km"] = self._travel_km(last_played, current_coords)
        return out

    def _last_played(self, priors: tuple[MatchRow, ...]) -> MatchRow | None:
        """The most recent prior match the player actually contested (walkovers
        excluded per C14; retirements included). `priors` is chronological, so scan
        from the end. None when every prior was a walkover."""
        for m in reversed(priors):
            if not m.walkover:
                return m
        return None

    def _match_weight(self, m: MatchRow) -> float:
        """C14 fatigue weight for one match: walkover → 1.0 only if
        `walkover_counts_as_match` (else 0.0); retirement → `retirement_fatigue_weight`
        only if `retirement_counts_as_match` (else 0.0); a completed match → 1.0."""
        if m.walkover:
            return 1.0 if self._walkover_counts else 0.0
        if m.retired:
            return self._retirement_weight if self._retirement_counts else 0.0
        return 1.0

    def _matches_count(self, window: list[MatchRow]) -> float:
        """Weighted count of window matches (a genuine 0.0 when the window is empty
        or holds only non-counting walkovers)."""
        return float(sum(self._match_weight(m) for m in window))

    def _minutes_sum(self, window: list[MatchRow]) -> float | None:
        """Weighted Σ `minutes` over counting window matches. NULL when a counting
        match lacks `minutes` (pre-1991 absence) — the load can't be completed.
        Non-counting matches (weight 0) are skipped and never trigger NULL; an empty
        window is a genuine 0.0 (the player rested)."""
        total = 0.0
        for m in window:
            weight = self._match_weight(m)
            if weight <= 0:
                continue
            if m.minutes is None:
                return None
            total += weight * m.minutes
        return total

    def _travel_km(
        self, last_played: MatchRow | None, current_coords: tuple[float, float] | None
    ) -> float | None:
        """Haversine km between the last-played venue and the current venue. NULL
        when there is no last match, no current coords, or the last venue's
        coordinates are unresolved."""
        if last_played is None or current_coords is None:
            return None
        last_coords = self._coords_for_match(last_played)
        if last_coords is None:
            return None
        return _haversine_km(*current_coords, *last_coords)

    # -- venue coordinate resolution (IO, memoized) ------------------------
    def _coords_for_match(self, m: MatchRow) -> tuple[float, float] | None:
        """The (lat, lon) of `m`'s tournament venue, or None when the tournament,
        its venue, or the coordinates are unresolved. A `StorageError` degrades to
        None (travel → NULL), never raised (§M8)."""
        venue_id = self._venue_id_for_tournament(m)
        if venue_id is None:
            return None
        return self._coords_for_venue(venue_id)

    def _venue_id_for_tournament(self, m: MatchRow) -> int | None:
        """`tournaments.venue_id` for `m`'s tournament (memoized per run). A
        `StorageError` returns None WITHOUT caching (transient — must not poison the
        cache); a clean None (no tournament / no venue) IS cached."""
        cached = self._tournament_venue_cache.get(m.tournament_id, _UNSET)
        if cached is not _UNSET:
            return cached  # type: ignore[return-value]
        try:
            tournament = self._tournament_repo.get(m.tournament_id)
        except StorageError as exc:
            _logger.warning(
                "fatigue_tournament_read_failed",
                match_id=m.match_id,
                tournament_id=m.tournament_id,
                cause=f"{type(exc).__name__}: {redact_text(str(exc))}",
            )
            return None
        venue_id = tournament.venue_id if tournament is not None else None
        self._tournament_venue_cache[m.tournament_id] = venue_id
        return venue_id

    def _coords_for_venue(self, venue_id: int | None) -> tuple[float, float] | None:
        """The (lat, lon) for a venue, or None when the venue id is missing, the
        venue is unresolved, or either coordinate is absent (memoized per run). A
        `StorageError` returns None WITHOUT caching; a clean None-from-data IS cached."""
        if venue_id is None:
            return None
        cached = self._venue_coords_cache.get(venue_id, _UNSET)
        if cached is not _UNSET:
            return cached  # type: ignore[return-value]
        try:
            venue = self._venue_repo.get(venue_id)
        except StorageError as exc:
            _logger.warning(
                "fatigue_venue_read_failed",
                venue_id=venue_id,
                cause=f"{type(exc).__name__}: {redact_text(str(exc))}",
            )
            return None
        coords = (
            None
            if venue is None or venue.latitude is None or venue.longitude is None
            else (venue.latitude, venue.longitude)
        )
        self._venue_coords_cache[venue_id] = coords
        return coords
