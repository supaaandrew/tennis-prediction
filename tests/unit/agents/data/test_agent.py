"""DataAgent control-flow tests (P7).

These exercise REAL control flow — step ordering, per-adapter fault isolation,
the staleness pre-flight halt, clock pinning, build-once resolver reuse, and the
§L5 weather-gap rule — using programmable fake adapters, not happy-path mocks.

Result objects are the REAL adapter dataclasses (SeasonResult / FetchResult /
ForecastResult) so the `.complete` semantics under test match production.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from tennis.adapters.atp_scraper.adapter import FetchResult as ScraperResult
from tennis.adapters.odds.adapter import FetchResult as OddsResult
from tennis.adapters.owm.adapter import ForecastResult
from tennis.adapters.sackmann.adapter import SeasonResult
from tennis.agents.data import DataAgent
from tennis.core import ids
from tennis.core.clock import Clock, FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import AgentContext
from tennis.core.errors import SackmannStalenessError
from tennis.storage.postgres.rows import VenueRow

_AS_OF = datetime(2026, 5, 24, 6, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[4]
    return load_config(root / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Fakes — record call order; programmable results / exceptions.
# ---------------------------------------------------------------------------
class _Factory:
    """Records (clock, run_id) per call and returns a fixed instance. Lets a
    test assert build-once (call count) and clock pinning (the clock passed)."""

    def __init__(self, instance: object) -> None:
        self.instance = instance
        self.calls: list[tuple[Clock, UUID]] = []

    def __call__(self, clock: Clock, run_id: UUID) -> object:
        self.calls.append((clock, run_id))
        return self.instance


class _FakeSackmann:
    def __init__(
        self,
        *,
        calls: list[tuple[str, str]],
        staleness: Exception | None = None,
        players: int = 5,
        season_result: Callable[[int], SeasonResult] | None = None,
    ) -> None:
        self._calls = calls
        self._staleness = staleness
        self._players = players
        self._season_result = season_result or (
            lambda y: SeasonResult(
                year=y, skipped=False, matches_ingested=1, conflicts_flagged=0
            )
        )
        self._players_registered = False

    def check_staleness(self) -> None:
        self._calls.append(("sackmann", "check_staleness"))
        if self._staleness is not None:
            raise self._staleness

    def check_pin_commit(self) -> None:
        self._calls.append(("sackmann", "check_pin_commit"))

    def ingest_players(self) -> int:
        self._calls.append(("sackmann", "ingest_players"))
        self._players_registered = True  # populates the in-memory resolver
        return self._players

    def ingest_season(self, year: int) -> SeasonResult:
        self._calls.append(("sackmann", "ingest_season"))
        # Build-once guard: ingest_season resolves against the resolver that
        # ingest_players populated. A fresh per-step adapter would have an empty
        # resolver — fail loudly so T10 catches a regression.
        if not self._players_registered:
            raise AssertionError("ingest_season ran before ingest_players (empty resolver)")
        return self._season_result(year)

    def ingest_rankings(self) -> int:
        self._calls.append(("sackmann", "ingest_rankings"))
        return 0


class _FakeScraper:
    def __init__(
        self, *, calls: list[tuple[str, str]],
        result: ScraperResult | None = None, exc: Exception | None = None,
    ) -> None:
        self._calls = calls
        self._exc = exc
        self._result = result or ScraperResult(
            tournaments_processed=1, matches_processed=2, matches_written=2,
            matches_skipped=0, conflicts_flagged=0, failures=0,
        )

    def fetch(self) -> ScraperResult:
        self._calls.append(("scraper", "fetch"))
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeOdds:
    def __init__(
        self, *, calls: list[tuple[str, str]],
        result: OddsResult | None = None, exc: Exception | None = None,
    ) -> None:
        self._calls = calls
        self._exc = exc
        self._result = result or OddsResult(
            events_processed=3, events_skipped=0, snapshots_inserted=6,
            matches_touched=3, failures=0,
        )

    def fetch_upcoming(self) -> OddsResult:
        self._calls.append(("odds", "fetch_upcoming"))
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeOwm:
    def __init__(
        self, *, calls: list[tuple[str, str]],
        result: ForecastResult | None = None, exc: Exception | None = None,
    ) -> None:
        self._calls = calls
        self._exc = exc
        self._result = result or ForecastResult(
            venues_processed=0, venues_skipped=0, observations_upserted=0,
            horizon_skipped=0, failures=0,
        )
        self.received_venue_ids: list[int] | None = None

    def fetch_forecasts(self, venue_ids: list[int]) -> ForecastResult:
        self._calls.append(("owm", "fetch_forecasts"))
        self.received_venue_ids = venue_ids
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeVenues:
    def __init__(self, rows: list[VenueRow]) -> None:
        self._rows = list(rows)
        self.upserted: list[VenueRow] = []

    def list_all(self) -> list[VenueRow]:
        return list(self._rows)

    def upsert(self, row: VenueRow) -> VenueRow:
        # Record + append so a subsequent list_all() sees it — lets T21 prove the
        # geocoding pass ran BEFORE the OWM enumeration.
        self.upserted.append(row)
        self._rows.append(row)
        return row


def _venue(vid: int, *, coords: bool) -> VenueRow:
    return VenueRow(
        venue_id=vid, city=f"city-{vid}", country_code="XX",
        latitude=51.5 if coords else None,
        longitude=-0.1 if coords else None,
    )


def _ctx(
    config: AppConfig, *,
    clock: Clock | None = None,
    heartbeat: Callable[[], None] | None = None,
    run_id: UUID | None = None,
) -> AgentContext:
    return AgentContext(
        run_id=run_id or uuid4(),
        as_of=_AS_OF,
        config=config,
        db=None,
        clock=clock or FrozenClock(_AS_OF),
        logger=None,
        heartbeat=heartbeat or (lambda: None),
    )


# A path guaranteed not to exist, so the §L11 geocoding pass is a clean no-op in
# every test that does not explicitly exercise it (warning + continue).
_NO_COORDS = Path(__file__).resolve().parent / "__nonexistent_venue_coords__.yaml"


def _build(
    config: AppConfig, *,
    sackmann: _FakeSackmann, scraper: _FakeScraper,
    odds: _FakeOdds, owm: _FakeOwm, venues: _FakeVenues,
    venue_coords_path: str | Path = _NO_COORDS,
    scraper_required: bool = True,
) -> tuple[DataAgent, dict[str, _Factory]]:
    factories = {
        "sackmann": _Factory(sackmann),
        "scraper": _Factory(scraper),
        "odds": _Factory(odds),
        "owm": _Factory(owm),
    }
    agent = DataAgent(
        config=config,
        sackmann_factory=factories["sackmann"],  # type: ignore[arg-type]
        scraper_factory=factories["scraper"],  # type: ignore[arg-type]
        odds_factory=factories["odds"],  # type: ignore[arg-type]
        owm_factory=factories["owm"],  # type: ignore[arg-type]
        venues=venues,  # type: ignore[arg-type]
        venue_coords_path=venue_coords_path,
        scraper_required=scraper_required,
    )
    return agent, factories


def _all_complete(calls: list[tuple[str, str]]) -> dict[str, object]:
    return dict(
        sackmann=_FakeSackmann(calls=calls),
        scraper=_FakeScraper(calls=calls),
        odds=_FakeOdds(calls=calls),
        owm=_FakeOwm(calls=calls),
    )


# ---------------------------------------------------------------------------
# T1 — step ordering (§J2 + inlined Sackmann order + preflight first)
# ---------------------------------------------------------------------------
class TestOrdering:
    def test_runs_scraper_then_sackmann_then_odds_then_owm(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            **_all_complete(calls),  # type: ignore[arg-type]
        )
        result = agent.run(_ctx(config))

        assert result.ok is True
        # preflight staleness check runs before any adapter
        assert calls[0] == ("sackmann", "check_staleness")
        # §J2 data-write order
        assert calls.index(("scraper", "fetch")) < calls.index(("sackmann", "ingest_players"))
        assert calls.index(("sackmann", "ingest_rankings")) < calls.index(("odds", "fetch_upcoming"))
        assert calls.index(("odds", "fetch_upcoming")) < calls.index(("owm", "fetch_forecasts"))
        # inlined Sackmann order
        i = lambda m: calls.index(("sackmann", m))  # noqa: E731
        assert i("check_pin_commit") < i("ingest_players") < i("ingest_season") < i("ingest_rankings")


# ---------------------------------------------------------------------------
# T2 / T3 — pre-flight halts: nothing else runs
# ---------------------------------------------------------------------------
class TestPreflightHalt:
    def test_staleness_halts_with_code_and_no_other_adapter(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        agent, factories = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls, staleness=SackmannStalenessError(
                last_pulled_at="2026-05-01T00:00:00+00:00", max_staleness_days=3)),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        assert result.ok is False
        assert [e.code for e in result.errors] == ["staleness_halt"]
        assert result.metrics == {}
        # no other adapter was even constructed
        assert factories["scraper"].calls == []
        assert factories["odds"].calls == []
        assert factories["owm"].calls == []

    def test_non_staleness_preflight_error_halts(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        agent, factories = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls, staleness=FileNotFoundError("mirror gone")),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        assert result.ok is False
        assert [e.code for e in result.errors] == ["preflight_error"]
        assert factories["scraper"].calls == []


# ---------------------------------------------------------------------------
# T4 / T5 / T19 — fault isolation: one adapter fails, the rest still run
# ---------------------------------------------------------------------------
class TestFaultIsolation:
    def test_odds_failure_count_downgrades_but_others_run(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        odds = _FakeOdds(calls=calls, result=OddsResult(
            events_processed=2, events_skipped=0, snapshots_inserted=1,
            matches_touched=1, failures=1))  # complete=False, no exception
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=odds,
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        assert result.ok is False
        assert result.errors == ()  # a result.complete=False is NOT an AgentError
        assert result.metrics["odds"]["complete"] is False
        # every adapter still ran
        assert set(result.metrics) == {"sackmann", "atp_scraper", "odds", "owm"}

    def test_adapter_exception_captured_others_run(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls, exc=RuntimeError("boom")),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        assert result.ok is False
        assert [e.code for e in result.errors] == ["atp_scraper_error"]
        assert result.metrics["atp_scraper"]["complete"] is False
        # sackmann / odds / owm still ran despite the scraper blowing up
        assert ("odds", "fetch_upcoming") in calls
        assert ("owm", "fetch_forecasts") in calls

    def test_odds_down_no_throw_signals_via_complete(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        odds = _FakeOdds(calls=calls, result=OddsResult(
            events_processed=0, events_skipped=0, snapshots_inserted=0,
            matches_touched=0, failures=2))
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=odds,
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        assert result.ok is False
        assert result.errors == ()  # never threw — signal is result.complete
        assert result.metrics["odds"]["complete"] is False


# ---------------------------------------------------------------------------
# §S6a — scraper optional for training (Cloudflare-blocked atptour.com must not
# gate model training, which consumes Sackmann `final` history only)
# ---------------------------------------------------------------------------
class TestScraperOptional:
    def test_required_false_tolerates_scraper_incomplete(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        # Scraper "succeeds" with failures>0 (the 403 path: adapter swallows the
        # block into failures, returns complete=False, no exception).
        scraper = _FakeScraper(calls=calls, result=ScraperResult(
            tournaments_processed=0, matches_processed=0, matches_written=0,
            matches_skipped=0, conflicts_flagged=0, failures=1))
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls), scraper=scraper,
            odds=_FakeOdds(calls=calls), owm=_FakeOwm(calls=calls),
            scraper_required=False,
        )
        result = agent.run(_ctx(config))

        assert result.ok is True  # Data succeeds despite the blocked scraper
        assert result.errors == ()
        assert result.metrics["atp_scraper"]["complete"] is False
        assert result.metrics["atp_scraper"]["required"] is False

    def test_required_false_still_surfaces_unexpected_scraper_exception(
        self, config: AppConfig
    ) -> None:
        # §S6a: scraper_required=False relaxes ONLY the operational-completeness
        # gate. An UNEXPECTED exception (a real bug — bad DI, parser regression)
        # must STILL surface as an AgentError so it is not masked during
        # training. The expected Cloudflare 403 never raises (the adapter absorbs
        # it into complete=False) — that path is covered above.
        calls: list[tuple[str, str]] = []
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls, exc=AttributeError("parser regression")),
            odds=_FakeOdds(calls=calls), owm=_FakeOwm(calls=calls),
            scraper_required=False,
        )
        result = agent.run(_ctx(config))

        assert result.ok is False  # the bug is NOT masked, even when optional
        assert [e.code for e in result.errors] == ["atp_scraper_error"]
        assert result.metrics["atp_scraper"]["complete"] is False

    def test_required_false_does_not_mask_other_adapter_failure(
        self, config: AppConfig
    ) -> None:
        calls: list[tuple[str, str]] = []
        odds = _FakeOdds(calls=calls, result=OddsResult(
            events_processed=0, events_skipped=0, snapshots_inserted=0,
            matches_touched=0, failures=1))  # a real downstream failure
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),  # clean scraper isolates the odds failure
            odds=odds, owm=_FakeOwm(calls=calls),
            scraper_required=False,
        )
        result = agent.run(_ctx(config))

        assert result.ok is False  # odds failure still downgrades Data

    def test_required_true_default_still_gates_on_scraper(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        scraper = _FakeScraper(calls=calls, result=ScraperResult(
            tournaments_processed=0, matches_processed=0, matches_written=0,
            matches_skipped=0, conflicts_flagged=0, failures=1))
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls), scraper=scraper,
            odds=_FakeOdds(calls=calls), owm=_FakeOwm(calls=calls),
        )  # scraper_required defaults True
        result = agent.run(_ctx(config))

        assert result.ok is False  # default behavior preserved (daily run)
        assert result.metrics["atp_scraper"]["required"] is True


# ---------------------------------------------------------------------------
# T6 / T6b / T7 — weather venue-coords gap (§L5)
# ---------------------------------------------------------------------------
class TestWeatherGap:
    def test_empty_venues_is_success_with_warning(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.data.agent._logger", mock_logger)
        calls: list[tuple[str, str]] = []
        owm = _FakeOwm(calls=calls)
        agent, _ = _build(
            config, venues=_FakeVenues([_venue(1, coords=False)]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=owm,
        )
        result = agent.run(_ctx(config))

        assert result.ok is True
        assert result.metrics["owm"]["complete"] is True
        assert owm.received_venue_ids == []  # no coord-bearing venues passed
        mock_logger.warning.assert_any_call(
            "owm_no_venues_v1_expected", venues_total=1, observations_upserted=0
        )

    def test_only_coord_bearing_venues_passed(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        owm = _FakeOwm(calls=calls)
        agent, _ = _build(
            config,
            venues=_FakeVenues([_venue(10, coords=True), _venue(20, coords=False)]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=owm,
        )
        agent.run(_ctx(config))

        assert owm.received_venue_ids == [10]

    def test_coord_venues_but_zero_observations_downgrades(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        owm = _FakeOwm(calls=calls, result=ForecastResult(
            venues_processed=1, venues_skipped=0, observations_upserted=0,
            horizon_skipped=0, failures=0))
        agent, _ = _build(
            config, venues=_FakeVenues([_venue(10, coords=True)]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=owm,
        )
        result = agent.run(_ctx(config))

        assert result.metrics["owm"]["complete"] is False
        assert result.ok is False
        assert result.errors == ()  # a "did-nothing" downgrade is not an error


# ---------------------------------------------------------------------------
# Secret redaction (Codex HIGH fix) — exception text never leaks into cause
# ---------------------------------------------------------------------------
class TestSecretRedaction:
    def test_api_key_in_exception_is_redacted_in_agent_error_cause(
        self, config: AppConfig
    ) -> None:
        calls: list[tuple[str, str]] = []
        fake_key = "sk_live_SUPERSECRET1234567890"
        # A transport-style error whose message carries a credential-bearing URL.
        boom = RuntimeError(
            f"401 Unauthorized for url "
            f"https://api.the-odds-api.com/v4/sports?apiKey={fake_key}&regions=us"
        )
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls, exc=boom),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        scraper_errors = [e for e in result.errors if e.code == "atp_scraper_error"]
        assert scraper_errors, "scraper exception should produce an AgentError"
        cause = scraper_errors[0].cause or ""
        # The raw secret must not survive into the cause (which is serialized
        # into pipeline_runs.error JSONB and logged).
        assert fake_key not in cause
        assert "***" in cause  # redaction actually fired
        # Type name is preserved for triage; no raw repr leak.
        assert cause.startswith("RuntimeError")
        # And nowhere in the whole serialized error set.
        assert fake_key not in repr(result.errors)


# ---------------------------------------------------------------------------
# T8 — heartbeat cadence
# ---------------------------------------------------------------------------
class TestHeartbeat:
    def test_emits_one_beat_before_each_step(self, config: AppConfig) -> None:
        beats = {"n": 0}
        calls: list[tuple[str, str]] = []
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            **_all_complete(calls),  # type: ignore[arg-type]
        )
        agent.run(_ctx(config, heartbeat=lambda: beats.__setitem__("n", beats["n"] + 1)))

        # one beat before each of the four adapter steps
        assert beats["n"] == 4


# ---------------------------------------------------------------------------
# T9 / T10 — clock pinning + build-once resolver reuse
# ---------------------------------------------------------------------------
class TestClockAndBuildOnce:
    def test_all_adapters_built_with_pinned_clock_and_run_id(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        run_id = uuid4()
        agent, factories = _build(
            config, venues=_FakeVenues([]),
            **_all_complete(calls),  # type: ignore[arg-type]
        )
        agent.run(_ctx(config, run_id=run_id))

        for name, factory in factories.items():
            assert factory.calls, f"{name} factory not invoked"
            clock, passed_run_id = factory.calls[0]
            assert clock.now() == _AS_OF, f"{name} not pinned to as_of"
            assert passed_run_id == run_id, f"{name} got wrong run_id"

    def test_sackmann_built_once_and_resolver_reused(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        agent, factories = _build(
            config, venues=_FakeVenues([]),
            **_all_complete(calls),  # type: ignore[arg-type]
        )
        result = agent.run(_ctx(config))

        # built exactly once — preflight + ingest share the instance (resolver state)
        assert len(factories["sackmann"].calls) == 1
        assert result.ok is True  # ingest_season never hit the empty-resolver guard


# ---------------------------------------------------------------------------
# T18 — Sackmann watermark skip counted as skipped, not ingested (§L6)
# ---------------------------------------------------------------------------
class TestSackmannSkip:
    def test_skipped_seasons_excluded_from_processed_count(self, config: AppConfig) -> None:
        calls: list[tuple[str, str]] = []
        sackmann = _FakeSackmann(
            calls=calls,
            season_result=lambda y: SeasonResult(
                year=y, skipped=True, matches_ingested=0, conflicts_flagged=0),
        )
        agent, _ = _build(
            config, venues=_FakeVenues([]),
            sackmann=sackmann,
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
        )
        result = agent.run(_ctx(config))

        assert result.metrics["sackmann"]["seasons_processed"] == 0
        assert result.metrics["sackmann"]["matches_ingested"] == 0
        assert result.metrics["sackmann"]["complete"] is True  # skips don't fail
        assert result.ok is True


# ---------------------------------------------------------------------------
# T21 — venue geocoding pass (§L11): YAML -> idempotent venue upserts, before OWM
# ---------------------------------------------------------------------------
_VENUE_YAML = """\
- slug: wimbledon
  city: London
  country_code: GBR
  lat: 51.4344
  lon: -0.2139
  altitude_m: null
- slug: queens-club
  city: London
  country_code: GBR
  lat: 51.4847
  lon: -0.213
  altitude_m: null
- slug: gstaad
  city: Gstaad
  country_code: CHE
  lat: 46.4722
  lon: 7.2887
  altitude_m: 1050
"""


class TestVenueGeocodingPass:
    def test_coords_loaded_and_upserted_before_owm(
        self, config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.data.agent._logger", mock_logger)
        yaml_path = tmp_path / "venue_coords.yaml"
        yaml_path.write_text(_VENUE_YAML, encoding="utf-8")

        calls: list[tuple[str, str]] = []
        # Coord-bearing venues now exist, so OWM must actually upsert observations
        # (obs>0) for the run to be 'succeeded' — the §L5 empty path no longer applies.
        owm = _FakeOwm(calls=calls, result=ForecastResult(
            venues_processed=2, venues_skipped=0, observations_upserted=4,
            horizon_skipped=0, failures=0))
        venues = _FakeVenues([])  # empty until the geocoding pass populates it
        agent, _ = _build(
            config, venues=venues,
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=owm,
            venue_coords_path=yaml_path,
        )
        result = agent.run(_ctx(config))

        # The London collision collapses to ONE venue (first wins -> Wimbledon
        # coords); Gstaad is distinct -> 2 upserts for 3 YAML entries.
        london_id = ids.venue_id(city="London", country_code="GBR")
        gstaad_id = ids.venue_id(city="Gstaad", country_code="CHE")
        assert [v.venue_id for v in venues.upserted] == [london_id, gstaad_id]
        london = venues.upserted[0]
        assert (london.city, london.country_code) == ("London", "GBR")
        assert (london.latitude, london.longitude) == (51.4344, -0.2139)  # first wins
        assert london.altitude_m is None
        gstaad = venues.upserted[1]
        assert (gstaad.latitude, gstaad.longitude, gstaad.altitude_m) == (46.4722, 7.2887, 1050)

        # Proves ordering: OWM enumerated AFTER the upserts, so it saw both venues.
        assert owm.received_venue_ids == [london_id, gstaad_id]
        mock_logger.info.assert_any_call("venue_geocoding_pass_complete", venues_upserted=2)
        assert result.ok is True

    def test_missing_yaml_warns_and_continues(
        self, config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.data.agent._logger", mock_logger)
        missing = tmp_path / "does_not_exist.yaml"

        calls: list[tuple[str, str]] = []
        venues = _FakeVenues([])
        agent, _ = _build(
            config, venues=venues,
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            venue_coords_path=missing,
        )
        # Must NOT raise; the run completes normally on the §L5 empty-venues path.
        result = agent.run(_ctx(config))

        assert venues.upserted == []
        mock_logger.warning.assert_any_call(
            "venue_coords_load_failed", path=str(missing), reason="missing"
        )
        assert result.ok is True  # missing coords degrades to §L5, not a failure


# ---------------------------------------------------------------------------
# §T-2 sidecar wiring — calendar, rankings, pre-fetch tier-id canary
# ---------------------------------------------------------------------------
from dataclasses import dataclass  # noqa: E402 — local-only


@dataclass(frozen=True)
class _GeoResult:
    venues_upserted: int
    failures: int


@dataclass(frozen=True)
class _RankResult:
    rows_upserted: int
    rows_skipped: int
    failures: int


class _FakeCalendar:
    def __init__(self, *, result: _GeoResult | None = None, exc: Exception | None = None) -> None:
        self._result = result or _GeoResult(venues_upserted=3, failures=0)
        self._exc = exc
        self.populate_calls: list[int] = []

    def populate(self, *, year: int) -> _GeoResult:
        self.populate_calls.append(year)
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeRankings:
    def __init__(self, *, result: _RankResult | None = None, exc: Exception | None = None) -> None:
        self._result = result or _RankResult(rows_upserted=100, rows_skipped=4, failures=0)
        self._exc = exc
        self.populate_calls: list[date] = []

    def populate(self, *, ranking_date: date) -> _RankResult:
        self.populate_calls.append(ranking_date)
        if self._exc is not None:
            raise self._exc
        return self._result


def _build_with_sidecars(
    config: AppConfig, *,
    sackmann: _FakeSackmann, scraper: _FakeScraper,
    odds: _FakeOdds, owm: _FakeOwm, venues: _FakeVenues,
    calendar: _FakeCalendar | None = None,
    rankings: _FakeRankings | None = None,
    pre_fetch_check: Callable[[], None] | None = None,
    venue_coords_path: str | Path = _NO_COORDS,
    scraper_required: bool = True,
) -> tuple[DataAgent, dict[str, _Factory]]:
    factories = {
        "sackmann": _Factory(sackmann),
        "scraper": _Factory(scraper),
        "odds": _Factory(odds),
        "owm": _Factory(owm),
    }
    agent = DataAgent(
        config=config,
        sackmann_factory=factories["sackmann"],  # type: ignore[arg-type]
        scraper_factory=factories["scraper"],  # type: ignore[arg-type]
        odds_factory=factories["odds"],  # type: ignore[arg-type]
        owm_factory=factories["owm"],  # type: ignore[arg-type]
        venues=venues,  # type: ignore[arg-type]
        venue_coords_path=venue_coords_path,
        scraper_required=scraper_required,
        matchstat_calendar_factory=(lambda: calendar) if calendar is not None else None,
        matchstat_rankings_factory=(lambda: rankings) if rankings is not None else None,
        pre_fetch_check=pre_fetch_check,
    )
    return agent, factories


# A date import in scope for the rankings/Monday tests.
from datetime import date  # noqa: E402


class TestT12CalendarStep:
    def test_calendar_runs_before_yaml_and_populates_metric(
        self, config: AppConfig, tmp_path: Path
    ) -> None:
        """Order assertion + metrics envelope: calendar runs INSIDE
        _step_geocode_venues BEFORE the §L11 YAML pass; success records
        calendar_venues_upserted + calendar_failures."""
        yaml_path = tmp_path / "venue_coords.yaml"
        yaml_path.write_text(_VENUE_YAML, encoding="utf-8")
        calls: list[tuple[str, str]] = []
        calendar = _FakeCalendar(result=_GeoResult(venues_upserted=7, failures=0))
        owm = _FakeOwm(calls=calls, result=ForecastResult(
            venues_processed=2, venues_skipped=0, observations_upserted=4,
            horizon_skipped=0, failures=0))
        venues = _FakeVenues([])
        agent, _ = _build_with_sidecars(
            config, venues=venues,
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=owm,
            calendar=calendar,
            venue_coords_path=yaml_path,
        )
        result = agent.run(_ctx(config))

        assert calendar.populate_calls == [_AS_OF.year]
        # Calendar produced metrics; status not affected by calendar.
        assert result.metrics["matchstat"]["calendar_venues_upserted"] == 7
        assert result.metrics["matchstat"]["calendar_failures"] == 0
        # YAML pass still ran (Wimbledon + Gstaad upserted in addition).
        assert {v.city for v in venues.upserted} == {"London", "Gstaad"}

    def test_calendar_factory_none_silently_skips(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.data.agent._logger", mock_logger)
        calls: list[tuple[str, str]] = []
        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            calendar=None,
        )
        result = agent.run(_ctx(config))
        assert "matchstat" not in result.metrics  # nothing wrote to the bucket
        mock_logger.info.assert_any_call(
            "matchstat_calendar_skipped", reason="no_factory"
        )

    def test_calendar_exception_warns_and_records_failure_no_agent_error(
        self, config: AppConfig
    ) -> None:
        """A sidecar exception must NEVER produce an AgentError — degrade
        to a structured WARN + `calendar_failures=1`. §L2 calc unchanged."""
        calls: list[tuple[str, str]] = []
        calendar = _FakeCalendar(exc=RuntimeError("matchstat unavailable"))
        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            calendar=calendar,
        )
        result = agent.run(_ctx(config))
        assert result.errors == ()  # NOT an AgentError
        assert result.metrics["matchstat"]["calendar_failures"] == 1
        assert result.metrics["matchstat"]["calendar_venues_upserted"] == 0
        # §L2 still proceeds; result.ok stays True (other steps clean).
        assert result.ok is True


class TestT13RankingsStep:
    def test_rankings_runs_between_odds_and_owm(self, config: AppConfig) -> None:
        """Step ordering: rankings fires AFTER odds, BEFORE owm.
        Asserts via the shared `calls` log so an accidental move of the
        rankings step (e.g. before odds or after OWM) fails this test."""
        calls: list[tuple[str, str]] = []
        # A wrapper rankings sidecar that appends to the SAME `calls` list,
        # so we can assert "odds before rankings before owm" via call order.
        class _OrderedRankings:
            def __init__(self) -> None:
                self.populate_calls: list[date] = []
            def populate(self, *, ranking_date: date):  # type: ignore[no-untyped-def]
                calls.append(("rankings", "populate"))
                self.populate_calls.append(ranking_date)
                return _RankResult(rows_upserted=100, rows_skipped=4, failures=0)
        rankings = _OrderedRankings()
        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            rankings=rankings,  # type: ignore[arg-type]
        )
        result = agent.run(_ctx(config))
        # Call ordering — the actual gate on §T13 placement.
        i_odds = calls.index(("odds", "fetch_upcoming"))
        i_rank = calls.index(("rankings", "populate"))
        i_owm = calls.index(("owm", "fetch_forecasts"))
        assert i_odds < i_rank < i_owm
        # Metrics envelope still populated.
        assert len(rankings.populate_calls) == 1
        assert result.metrics["matchstat"]["rankings_upserted"] == 100
        assert result.metrics["matchstat"]["rankings_skipped"] == 4

    def test_rankings_monday_derivation_from_non_monday_clock(
        self, config: AppConfig
    ) -> None:
        """_AS_OF is 2026-05-24 = Sunday; the ISO-week Monday is 2026-05-18."""
        calls: list[tuple[str, str]] = []
        rankings = _FakeRankings()
        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            rankings=rankings,
        )
        agent.run(_ctx(config))
        assert rankings.populate_calls == [date(2026, 5, 18)]

    def test_rankings_factory_none_skips_no_metric(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr("tennis.agents.data.agent._logger", mock_logger)
        calls: list[tuple[str, str]] = []
        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            rankings=None,
        )
        agent.run(_ctx(config))
        mock_logger.info.assert_any_call(
            "matchstat_rankings_skipped", reason="no_factory"
        )

    def test_rankings_exception_warns_and_records_failure(
        self, config: AppConfig
    ) -> None:
        calls: list[tuple[str, str]] = []
        rankings = _FakeRankings(exc=RuntimeError("matchstat 500"))
        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            rankings=rankings,
        )
        result = agent.run(_ctx(config))
        assert result.errors == ()
        assert result.metrics["matchstat"]["rankings_failures"] == 1
        assert result.metrics["matchstat"]["rankings_upserted"] == 0


class TestT5PreFetchCheck:
    def test_pre_fetch_check_runs_before_scraper_fetch(self, config: AppConfig) -> None:
        """The canary must fire BEFORE scraper.fetch() so a renumbered tier
        id never silently corrupts a slate."""
        calls: list[tuple[str, str]] = []
        order: list[str] = []

        def pre_check() -> None:
            order.append("pre_check")

        original_fetch = _FakeScraper.fetch

        def tracked_fetch(self):  # type: ignore[no-untyped-def]
            order.append("fetch")
            return original_fetch(self)

        scraper = _FakeScraper(calls=calls)
        scraper.fetch = tracked_fetch.__get__(scraper, _FakeScraper)  # type: ignore[assignment]

        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=scraper,
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            pre_fetch_check=pre_check,
        )
        result = agent.run(_ctx(config))
        assert order == ["pre_check", "fetch"]
        assert result.ok is True

    def test_pre_fetch_check_failure_drops_data_to_partial(
        self, config: AppConfig
    ) -> None:
        """A pre-fetch `MatchstatTierMismatchError` lands in the §L2 _capture
        path under its OWN distinct code, NOT the generic `atp_scraper_error`
        — so operators can tell a tier renumbering canary apart from a
        scraper bug. Data drops to partial; other steps still run."""
        from tennis.core.errors import MatchstatTierMismatchError

        calls: list[tuple[str, str]] = []

        def pre_check() -> None:
            raise MatchstatTierMismatchError(
                "matchstat tier id 2 maps to 'Renamed Tier', expected one of [...]"
            )

        agent, _ = _build_with_sidecars(
            config, venues=_FakeVenues([]),
            sackmann=_FakeSackmann(calls=calls),
            scraper=_FakeScraper(calls=calls),
            odds=_FakeOdds(calls=calls),
            owm=_FakeOwm(calls=calls),
            pre_fetch_check=pre_check,
        )
        result = agent.run(_ctx(config))
        assert result.ok is False
        codes = {e.code for e in result.errors}
        assert codes == {"matchstat_tier_mismatch"}  # distinct from generic scraper bug
        # The scraper.fetch() was NOT called (pre-check raised first).
        assert ("scraper", "fetch") not in calls
        # Other adapters still ran.
        assert ("sackmann", "ingest_players") in calls
        assert ("odds", "fetch_upcoming") in calls
