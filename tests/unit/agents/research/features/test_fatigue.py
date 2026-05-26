"""Tests for the Fatigue family (R7) — `features/fatigue.py`.

Exercises `FatigueExtractor`: rest-days from the last *played* match (retirements
included, walkovers excluded per C14), the 7/14-day weighted match counts and
weighted minutes sums, the pre-1991 minutes NULL-by-absence rule, the haversine
travel distance and its missing-coord/venue NULL paths, the debut/empty-history
all-NULL path, the catalog-faithful "best_of does not weight load" guarantee,
config-driven C14 knobs, `StorageError`→NULL degradation for travel, dtypes, and
the §M7 lockstep round-trip against the seeded `feature_specs` rows.

All in-memory; no Docker.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.fatigue import (
    FATIGUE_FAMILY,
    FATIGUE_FEATURE_KEYS,
    FatigueExtractor,
    _haversine_km,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import StorageError
from tennis.storage.postgres.rows import MatchRow, Surface, TournamentRow, VenueRow

_P1 = 10
_P2 = 20
_OPP = 99  # prior-match opponent (kept off _P2 so _P2 stays a debut by default)
_AS_OF = datetime(2024, 6, 20, 12, 0, tzinfo=UTC)

_CUR_VENUE = 500  # London
_LAST_VENUE = 600  # Paris (reached via tournament 60)
_LAST_TOURNAMENT = 60

# London / Paris coordinates — a known ~343 km great-circle separation.
_LONDON = (51.5074, -0.1278)
_PARIS = (48.8566, 2.3522)

_DEFAULT_VENUES = {
    _CUR_VENUE: VenueRow(venue_id=_CUR_VENUE, city="London", country_code="GB",
                         latitude=_LONDON[0], longitude=_LONDON[1]),
    _LAST_VENUE: VenueRow(venue_id=_LAST_VENUE, city="Paris", country_code="FR",
                          latitude=_PARIS[0], longitude=_PARIS[1]),
}
_DEFAULT_TOURNAMENTS = {
    _LAST_TOURNAMENT: TournamentRow(
        tournament_id=_LAST_TOURNAMENT, season=2024, slug="paris", name="Paris",
        tier="ATP500", surface="Hard", indoor=False, venue_id=_LAST_VENUE),
}


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeVenueRepo:
    def __init__(self, venues: dict[int, VenueRow], *, raise_for: set[int] = frozenset()):
        self._venues = venues
        self._raise_for = set(raise_for)
        self.calls = 0  # total get() invocations — for memoization assertions

    def get(self, venue_id: int) -> VenueRow | None:
        self.calls += 1
        if venue_id in self._raise_for:
            raise StorageError("venue repo boom: secret=shhh")
        return self._venues.get(venue_id)


class _FakeTournamentRepo:
    def __init__(self, tournaments: dict[int, TournamentRow],
                 *, raise_for: set[int] = frozenset()):
        self._tournaments = tournaments
        self._raise_for = set(raise_for)
        self.calls = 0

    def get(self, tournament_id: int) -> TournamentRow | None:
        self.calls += 1
        if tournament_id in self._raise_for:
            raise StorageError("tournament repo boom")
        return self._tournaments.get(tournament_id)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _prior(
    match_id: int,
    *,
    day: int,
    player: int = _P1,
    opponent: int = _OPP,
    tournament_id: int = _LAST_TOURNAMENT,
    minutes: int | None = None,
    retired: bool = False,
    walkover: bool = False,
    best_of: int | None = 3,
    hour: int = 12,
) -> MatchRow:
    """A historical `final` prior match for `player` in June 2024."""
    return MatchRow(
        match_id=match_id, tournament_id=tournament_id, round="R32",
        match_date=date(2024, 6, day), p1_id=player, p2_id=opponent,
        status="final", source="test", source_uid=f"m{match_id}",
        start_ts=datetime(2024, 6, day, hour, tzinfo=UTC),
        winner_id=player, best_of=best_of, minutes=minutes,
        retired=retired, walkover=walkover,
    )


def _index(*priors: MatchRow) -> MatchHistoryIndex:
    return MatchHistoryIndex.build(list(priors))


def _fctx(
    *,
    venue_id: int | None = _CUR_VENUE,
    p1: int = _P1,
    p2: int = _P2,
    as_of: datetime = _AS_OF,
    surface: Surface = "Hard",
) -> FeatureContext:
    match = MatchRow(
        match_id=999, tournament_id=900, round="R16",
        match_date=as_of.date(), p1_id=p1, p2_id=p2, status="scheduled",
        source="test", source_uid="cur", start_ts=as_of + timedelta(hours=24),
    )
    return FeatureContext(
        match=match, as_of_ts=as_of, feature_set="v1",
        surface=surface, indoor=False, venue_id=venue_id, tier="ATP500",
    )


def _extractor(
    config: AppConfig,
    *,
    index: MatchHistoryIndex,
    venue_repo: _FakeVenueRepo | None = None,
    tournament_repo: _FakeTournamentRepo | None = None,
) -> FatigueExtractor:
    return FatigueExtractor(
        history=index,
        tournament_repo=tournament_repo or _FakeTournamentRepo(dict(_DEFAULT_TOURNAMENTS)),
        venue_repo=venue_repo or _FakeVenueRepo(dict(_DEFAULT_VENUES)),
        config=config,
    )


# A "rich" P1 history: one played match (15th, 120m), one retirement (17th, 60m),
# one older played match (8th, 90m), one walkover (16th). With as_of=06-20 12:00 the
# 7d window opens 06-13 12:00 and the 14d window opens 06-06 12:00.
def _rich_index() -> MatchHistoryIndex:
    return _index(
        _prior(1, day=15, minutes=120),
        _prior(2, day=17, minutes=60, retired=True),
        _prior(3, day=8, minutes=90),
        _prior(4, day=16, walkover=True, minutes=None),
    )


def _flip(config: AppConfig, **fe_updates) -> AppConfig:
    fe = config.feature_engineering.model_copy(update=fe_updates)
    return config.model_copy(update={"feature_engineering": fe})


# ---------------------------------------------------------------------------
class TestHaversine:
    def test_known_distance_london_paris(self) -> None:
        # London↔Paris ≈ 343 km — validates the great-circle formula.
        km = _haversine_km(*_LONDON, *_PARIS)
        assert km == pytest.approx(343.6, abs=2.0)

    def test_zero_distance_same_point(self) -> None:
        assert _haversine_km(*_LONDON, *_LONDON) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self) -> None:
        assert _haversine_km(*_LONDON, *_PARIS) == pytest.approx(
            _haversine_km(*_PARIS, *_LONDON)
        )


# ---------------------------------------------------------------------------
class TestRestDays:
    def test_days_since_last_played(self, config: AppConfig) -> None:
        # last_played = retirement on the 17th (retirements count) → 06-20 − 06-17 = 3.
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        assert out["p1_rest_days"] == 3

    def test_walkover_excluded_from_last_match(self, config: AppConfig) -> None:
        # Most recent prior is a walkover (18th); the last *played* match is the 12th.
        idx = _index(
            _prior(1, day=12, minutes=90),
            _prior(2, day=18, walkover=True),
        )
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_rest_days"] == 8  # 06-20 − 06-12

    def test_retirement_counts_as_last_match(self, config: AppConfig) -> None:
        idx = _index(_prior(1, day=14, minutes=40, retired=True))
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_rest_days"] == 6  # 06-20 − 06-14

    def test_null_when_all_priors_walkovers(self, config: AppConfig) -> None:
        idx = _index(_prior(1, day=12, walkover=True))
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_rest_days"] is None


# ---------------------------------------------------------------------------
class TestMatchCounts:
    def test_weighted_counts_7d_and_14d(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        # 7d: 15th(1.0) + 17th retire(0.5) + 16th walkover(0) = 1.5
        assert out["p1_matches_last_7d"] == pytest.approx(1.5)
        # 14d: + 8th(1.0) = 2.5
        assert out["p1_matches_last_14d"] == pytest.approx(2.5)

    def test_zero_when_history_but_no_window_matches(self, config: AppConfig) -> None:
        # A real zero (rested), NOT NULL: player has history, just nothing recent.
        idx = _index(_prior(1, day=1, minutes=90))  # 06-01, outside 14d
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_matches_last_7d"] == 0.0
        assert out["p1_matches_last_14d"] == 0.0

    def test_window_lower_bound_inclusive(self, config: AppConfig) -> None:
        # A match exactly at the 7d lower edge (06-13 12:00) is included.
        idx = _index(_prior(1, day=13, hour=12, minutes=90))
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_matches_last_7d"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
class TestMinutes:
    def test_weighted_minutes_sums(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        # 7d: 120 + 0.5*60 = 150 (walkover excluded)
        assert out["p1_minutes_last_7d"] == pytest.approx(150.0)
        # 14d: + 90 = 240
        assert out["p1_minutes_last_14d"] == pytest.approx(240.0)

    def test_null_when_counting_window_match_lacks_minutes(self, config: AppConfig) -> None:
        # Pre-1991 NULL-by-absence: a played window match with minutes=None → NULL.
        idx = _index(
            _prior(1, day=15, minutes=120),
            _prior(2, day=16, minutes=None),  # played, no minutes
        )
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_minutes_last_7d"] is None
        # The count is unaffected — minutes absence does not suppress the count.
        assert out["p1_minutes_last_14d"] is None
        assert out["p1_matches_last_7d"] == pytest.approx(2.0)

    def test_walkover_missing_minutes_does_not_null(self, config: AppConfig) -> None:
        # A walkover has weight 0 → skipped → its missing minutes never triggers NULL.
        idx = _index(
            _prior(1, day=15, minutes=120),
            _prior(2, day=16, walkover=True, minutes=None),
        )
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_minutes_last_7d"] == pytest.approx(120.0)

    def test_zero_minutes_when_history_but_window_empty(self, config: AppConfig) -> None:
        idx = _index(_prior(1, day=1, minutes=90))  # outside both windows
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_minutes_last_7d"] == 0.0
        assert out["p1_minutes_last_14d"] == 0.0


# ---------------------------------------------------------------------------
class TestTravel:
    def test_haversine_between_last_and_current_venue(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        # last_played = 17th @ tournament 60 → Paris; current venue = London.
        assert out["p1_travel_km_since_last_match"] == pytest.approx(
            _haversine_km(*_LONDON, *_PARIS)
        )

    def test_null_when_current_venue_missing(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx(venue_id=None))
        assert out["p1_travel_km_since_last_match"] is None

    def test_null_when_current_venue_unresolved(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx(venue_id=12345))
        assert out["p1_travel_km_since_last_match"] is None

    def test_null_when_current_venue_lacks_coords(self, config: AppConfig) -> None:
        venues = dict(_DEFAULT_VENUES)
        venues[_CUR_VENUE] = VenueRow(venue_id=_CUR_VENUE, city="X", country_code="GB")
        repo = _FakeVenueRepo(venues)
        out = _extractor(config, index=_rich_index(), venue_repo=repo).extract(_fctx())
        assert out["p1_travel_km_since_last_match"] is None

    def test_null_when_last_tournament_has_no_venue(self, config: AppConfig) -> None:
        tournaments = {
            _LAST_TOURNAMENT: TournamentRow(
                tournament_id=_LAST_TOURNAMENT, season=2024, slug="x", name="X",
                tier="ATP500", surface="Hard", indoor=False, venue_id=None),
        }
        repo = _FakeTournamentRepo(tournaments)
        out = _extractor(config, index=_rich_index(), tournament_repo=repo).extract(_fctx())
        assert out["p1_travel_km_since_last_match"] is None

    def test_null_when_last_played_is_walkover_history_only(self, config: AppConfig) -> None:
        # No actual last match → no travel origin.
        idx = _index(_prior(1, day=12, walkover=True))
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_travel_km_since_last_match"] is None


# ---------------------------------------------------------------------------
class TestStorageErrorDegradation:
    def test_venue_storage_error_nulls_travel_only(self, config: AppConfig) -> None:
        # A StorageError reading the venue degrades travel to NULL; the non-IO keys
        # (rest_days, counts, minutes) are still computed.
        repo = _FakeVenueRepo(dict(_DEFAULT_VENUES), raise_for={_CUR_VENUE})
        out = _extractor(config, index=_rich_index(), venue_repo=repo).extract(_fctx())
        assert out["p1_travel_km_since_last_match"] is None
        assert out["p1_rest_days"] == 3
        assert out["p1_matches_last_7d"] == pytest.approx(1.5)
        assert out["p1_minutes_last_7d"] == pytest.approx(150.0)

    def test_tournament_storage_error_nulls_travel_only(self, config: AppConfig) -> None:
        repo = _FakeTournamentRepo(dict(_DEFAULT_TOURNAMENTS), raise_for={_LAST_TOURNAMENT})
        out = _extractor(config, index=_rich_index(), tournament_repo=repo).extract(_fctx())
        assert out["p1_travel_km_since_last_match"] is None
        assert out["p1_rest_days"] == 3


# ---------------------------------------------------------------------------
class TestMemoization:
    def test_venue_and_tournament_lookups_are_cached(self, config: AppConfig) -> None:
        # Both players' last matches resolve to the same tournament/venue, and the
        # current venue is shared — so within a run each unique id is read once, not
        # once per match/player (retires the §M18-style N+1).
        venue_repo = _FakeVenueRepo(dict(_DEFAULT_VENUES))
        tournament_repo = _FakeTournamentRepo(dict(_DEFAULT_TOURNAMENTS))
        idx = _index(
            _prior(1, day=15, player=_P1, minutes=120),
            _prior(2, day=14, player=_P2, opponent=88, minutes=90),
        )
        ext = _extractor(
            config, index=idx, venue_repo=venue_repo, tournament_repo=tournament_repo
        )
        ext.extract(_fctx())
        # current venue (500) + last venue (600, shared across both players via cache).
        assert venue_repo.calls == 2
        assert tournament_repo.calls == 1  # tournament 60 resolved once
        # A second match over the same venues hits the cache → zero new reads.
        ext.extract(_fctx())
        assert venue_repo.calls == 2
        assert tournament_repo.calls == 1

    def test_storage_error_is_not_cached(self, config: AppConfig) -> None:
        # A transient StorageError must NOT be cached as a permanent miss — the next
        # extract retries the read rather than returning a poisoned NULL.
        venue_repo = _FakeVenueRepo(dict(_DEFAULT_VENUES), raise_for={_CUR_VENUE})
        ext = _extractor(config, index=_rich_index(), venue_repo=venue_repo)
        ext.extract(_fctx())
        first = venue_repo.calls
        ext.extract(_fctx())
        assert venue_repo.calls > first  # re-attempted, not served from cache


# ---------------------------------------------------------------------------
class TestDebutAllNull:
    def test_no_priors_all_keys_null(self, config: AppConfig) -> None:
        # _P2 has no priors in the rich index → every p2 key NULL.
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        for key in (
            "p2_rest_days", "p2_matches_last_7d", "p2_matches_last_14d",
            "p2_minutes_last_7d", "p2_minutes_last_14d",
            "p2_travel_km_since_last_match",
        ):
            assert out[key] is None

    def test_empty_index_all_keys_null(self, config: AppConfig) -> None:
        out = _extractor(config, index=_index()).extract(_fctx())
        for key in FATIGUE_FEATURE_KEYS:
            assert out[key] is None


# ---------------------------------------------------------------------------
class TestPIT:
    def test_match_at_or_after_as_of_excluded(self, config: AppConfig) -> None:
        # Strict PIT: a prior whose instant == as_of (or later) is not visible.
        idx = _index(
            _prior(1, day=19, minutes=90),                 # strictly before
            _prior(2, day=20, hour=12, minutes=90),         # instant == as_of
        )
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_rest_days"] == 1  # only the 06-19 match is visible
        assert out["p1_matches_last_7d"] == pytest.approx(1.0)

    def test_both_players_computed_independently(self, config: AppConfig) -> None:
        idx = _index(
            _prior(1, day=15, player=_P1, minutes=120),
            _prior(2, day=10, player=_P2, opponent=88, minutes=90),
        )
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_rest_days"] == 5   # 06-20 − 06-15
        assert out["p2_rest_days"] == 10  # 06-20 − 06-10
        assert out["p1_minutes_last_7d"] == pytest.approx(120.0)
        assert out["p2_minutes_last_7d"] == 0.0  # 06-10 outside the 7d window


# ---------------------------------------------------------------------------
class TestBestOfNotUsed:
    def test_best_of_does_not_alter_fatigue(self, config: AppConfig) -> None:
        # Catalog-faithful: best_of is NOT a fatigue multiplier. Identical histories
        # differing only in best_of must yield identical fatigue output.
        bo3 = _index(_prior(1, day=15, minutes=120, best_of=3))
        bo5 = _index(_prior(1, day=15, minutes=120, best_of=5))
        out3 = _extractor(config, index=bo3).extract(_fctx())
        out5 = _extractor(config, index=bo5).extract(_fctx())
        assert out3 == out5

    def test_best_of_none_does_not_break(self, config: AppConfig) -> None:
        idx = _index(_prior(1, day=15, minutes=120, best_of=None))
        out = _extractor(config, index=idx).extract(_fctx())
        assert out["p1_minutes_last_7d"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
class TestC14ConfigDriven:
    def test_retirement_not_counted_when_disabled(self, config: AppConfig) -> None:
        cfg = _flip(config, retirement_counts_as_match=False)
        out = _extractor(cfg, index=_rich_index()).extract(_fctx())
        # 7d count drops the retirement: 15th(1.0) + 17th(0) + 16th wo(0) = 1.0
        assert out["p1_matches_last_7d"] == pytest.approx(1.0)
        # retirement minutes no longer contribute: only the 120m played match.
        assert out["p1_minutes_last_7d"] == pytest.approx(120.0)

    def test_walkover_counted_when_enabled(self, config: AppConfig) -> None:
        cfg = _flip(config, walkover_counts_as_match=True)
        out = _extractor(cfg, index=_rich_index()).extract(_fctx())
        # 7d count now adds the walkover at full weight: 1.0 + 0.5 + 1.0 = 2.5
        assert out["p1_matches_last_7d"] == pytest.approx(2.5)

    def test_retirement_weight_respected(self, config: AppConfig) -> None:
        cfg = _flip(config, retirement_fatigue_weight=0.25)
        out = _extractor(cfg, index=_rich_index()).extract(_fctx())
        # 7d count: 1.0 + 0.25 = 1.25
        assert out["p1_matches_last_7d"] == pytest.approx(1.25)
        # 7d minutes: 120 + 0.25*60 = 135
        assert out["p1_minutes_last_7d"] == pytest.approx(135.0)


# ---------------------------------------------------------------------------
class TestContract:
    def test_name_is_fatigue(self, config: AppConfig) -> None:
        ext = _extractor(config, index=_index())
        assert ext.name == FATIGUE_FAMILY == "fatigue"

    def test_implements_feature_extractor_protocol(self, config: AppConfig) -> None:
        assert isinstance(_extractor(config, index=_index()), FeatureExtractor)

    def test_feature_keys_has_twelve(self, config: AppConfig) -> None:
        assert len(_extractor(config, index=_index()).feature_keys()) == 12

    def test_feature_keys_equal_seeded_fatigue_rows(self, config: AppConfig) -> None:
        seeded = {row.feature_key for row in _REGISTRY["fatigue"]}
        assert set(_extractor(config, index=_index()).feature_keys()) == seeded

    def test_extract_always_emits_all_twelve_keys(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        assert set(out) == set(FATIGUE_FEATURE_KEYS)

    def test_dtypes_native_python(self, config: AppConfig) -> None:
        out = _extractor(config, index=_rich_index()).extract(_fctx())
        # rest_days is whole days (int, not bool).
        assert isinstance(out["p1_rest_days"], int) and not isinstance(
            out["p1_rest_days"], bool
        )
        assert isinstance(out["p1_matches_last_7d"], float)
        assert isinstance(out["p1_minutes_last_7d"], float)
        assert isinstance(out["p1_travel_km_since_last_match"], float)
