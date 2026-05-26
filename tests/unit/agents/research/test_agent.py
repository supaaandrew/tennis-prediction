"""ResearchAgent orchestration tests (R6a).

Exercises real control flow with in-memory fakes (no Docker): training vs
prediction scope, the §M12 windows guard, the decision-2 None-tournament
skip+dead-letter, per-match fault isolation, the C10 validate-before-write gate,
feature_specs seeding, the seven-family merge, and heartbeat emission. Uses the
REAL extractors via the default registry so the merged matrix is a faithful row;
the per-family math is covered by the family-level tests.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from tennis.agents.research import ResearchAgent
from tennis.agents.research import specs
from tennis.agents.research.context import FeatureContext
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.contracts import AgentContext
from tennis.core.errors import FeatureContractError
from tennis.storage.postgres.rows import MatchRow, TournamentRow

_NOW = datetime(2026, 5, 24, 6, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[4]
    return load_config(root / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------
def _final(match_id: int, *, tournament_id: int, match_date: date) -> MatchRow:
    """A historical `final` (start_ts NULL → historical PIT cut)."""
    return MatchRow(
        match_id=match_id, tournament_id=tournament_id, round="R16",
        match_date=match_date, p1_id=1, p2_id=2, status="final",
        source="test", source_uid=f"f{match_id}", start_ts=None,
        winner_id=1, match_date_source="sackmann",
    )


def _scheduled(match_id: int, *, tournament_id: int, start_ts: datetime) -> MatchRow:
    """An upcoming `scheduled` match (start_ts set → live PIT cut)."""
    return MatchRow(
        match_id=match_id, tournament_id=tournament_id, round="R16",
        match_date=start_ts.date(), p1_id=1, p2_id=2, status="scheduled",
        source="test", source_uid=f"s{match_id}", start_ts=start_ts,
        match_date_source="atp_scraper",
    )


_TOURNAMENT = TournamentRow(
    tournament_id=0, season=2026, slug="t", name="T",
    tier="GS", surface="Hard", indoor=False, venue_id=None,
)


# ---------------------------------------------------------------------------
# In-memory fakes — Protocol-faithful enough for the agent
# ---------------------------------------------------------------------------
class _FakeMatchRepo:
    def __init__(self, *, finals=(), scheduled=()):
        self._finals = list(finals)
        self._scheduled = list(scheduled)
        self.for_training_calls = 0
        self.for_prediction_calls = 0

    def for_training(self, *, season_start, season_end):
        self.for_training_calls += 1
        return list(self._finals)

    def for_prediction(self, *, as_of, lookforward_days):
        self.for_prediction_calls += 1
        return list(self._scheduled)


class _FakeEloRepo:
    def __init__(self, *, counts=None):
        self.inserts = []
        self._counts = dict(counts or {})
        self.career_counts_calls = 0

    def insert(self, row):
        self.inserts.append(row)
        return row

    def get_latest_before(self, *, player_id, surface, as_of_ts):
        return None  # cold start → 1500 fallback (H10)

    def career_match_counts(self):
        self.career_counts_calls += 1
        return dict(self._counts)


class _FakeTournamentRepo:
    def __init__(self, *, none_for=(), raise_for=()):
        self._none_for = set(none_for)
        self._raise_for = set(raise_for)

    def get(self, tournament_id):
        if tournament_id in self._raise_for:
            raise RuntimeError("tournament repo boom: secret=shhh")
        if tournament_id in self._none_for:
            return None
        return dataclasses.replace(_TOURNAMENT, tournament_id=tournament_id)

    def get_by_season_slug(self, *, season, slug):
        return None

    def upsert(self, row):
        return row


class _FakeRankingRepo:
    def get(self, *, player_id, ranking_date):
        return None

    def latest_before(self, *, player_id, on_or_before):
        return None

    def upsert(self, row):
        return row


class _FakeStatRepo:
    def get(self, *, match_id, player_id):
        return None

    def list_for_match(self, match_id):
        return []

    def upsert(self, row):
        return row


class _FakeWeatherRepo:
    def get(self, *, venue_id, observed_at, source):
        return None

    def nearest_at_or_before(self, *, venue_id, target_ts, source, max_age_hours):
        return None

    def upsert(self, row):
        return row


class _FakeVenueRepo:
    def get(self, venue_id):
        return None

    def get_by_city_country(self, *, city, country_code):
        return None

    def list_all(self):
        return []

    def upsert(self, row):
        return row


class _FakeFeatureSpecRepo:
    def __init__(self):
        self.specs: dict[tuple[str, int], object] = {}

    def get(self, *, feature_key, version):
        return self.specs.get((feature_key, version))

    def list_active(self, *, feature_set):
        return list(self.specs.values())

    def upsert(self, row):
        self.specs[(row.feature_key, row.version)] = row
        return row


class _FakeFeatureMatrixRepo:
    def __init__(self):
        self.rows = []

    def get(self, *, match_id, feature_set):
        return None

    def upsert(self, row):
        self.rows.append(row)
        return row

    def list_for_matches(self, *, match_ids, feature_set):
        return []


class _FakeDeadLetter:
    def __init__(self):
        self.rows = []

    def append(self, row):  # MUST never raise
        self.rows.append(row)

    def list_recent(self, *, limit=100):
        return list(self.rows)


class _Stub:
    """A stand-in extractor returning a fixed fragment (or raising)."""

    def __init__(self, name, fragment):
        self.name = name
        self._fragment = dict(fragment)

    def feature_keys(self):
        return tuple(self._fragment.keys())

    def extract(self, fctx: FeatureContext):
        return dict(self._fragment)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------
def _agent(config, *, mode, match_repo, elo_repo=None, tournament_repo=None,
           feature_spec_repo=None, feature_matrix_repo=None, dead_letter=None,
           extractor_factories=None):
    kwargs = dict(
        mode=mode, config=config, match_repo=match_repo,
        elo_repo=elo_repo or _FakeEloRepo(),
        tournament_repo=tournament_repo or _FakeTournamentRepo(),
        ranking_repo=_FakeRankingRepo(),
        stat_repo=_FakeStatRepo(),
        weather_repo=_FakeWeatherRepo(),
        venue_repo=_FakeVenueRepo(),
        feature_spec_repo=feature_spec_repo or _FakeFeatureSpecRepo(),
        feature_matrix_repo=feature_matrix_repo or _FakeFeatureMatrixRepo(),
        dead_letter=dead_letter or _FakeDeadLetter(),
    )
    if extractor_factories is not None:
        kwargs["extractor_factories"] = extractor_factories
    return ResearchAgent(**kwargs)


def _ctx(config) -> tuple[AgentContext, list[int]]:
    beats: list[int] = []
    ctx = AgentContext(
        run_id=uuid4(), as_of=_NOW, config=config, db=None,
        clock=FrozenClock(_NOW), logger=None, heartbeat=lambda: beats.append(1),
    )
    return ctx, beats


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------
class TestLineage:
    def test_declares_data_succeeded_precondition(self, config: AppConfig) -> None:
        agent = _agent(config, mode="training", match_repo=_FakeMatchRepo())
        assert agent.name == "research"
        pres = agent.lineage.preconditions
        assert len(pres) == 1
        assert pres[0].previous_agent == "data"
        assert pres[0].required_status == "succeeded"

    def test_rejects_unknown_mode(self, config: AppConfig) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            _agent(config, mode="backtest", match_repo=_FakeMatchRepo())


# ---------------------------------------------------------------------------
# §M12 windows guard
# ---------------------------------------------------------------------------
class TestWindowsGuard:
    def test_matching_windows_pass(self, config: AppConfig) -> None:
        # config.yaml windows_days == specs._FORM_WINDOWS → no raise.
        assert tuple(config.features.windows_days) == tuple(specs._FORM_WINDOWS)
        agent = _agent(config, mode="training", match_repo=_FakeMatchRepo())
        ctx, _ = _ctx(config)
        result = agent.run(ctx)  # empty scope, but the guard ran first
        assert result.ok is True

    def test_divergent_windows_raise_at_construction(self, config: AppConfig) -> None:
        bad_features = config.features.model_copy(update={"windows_days": (7, 14, 30)})
        bad = config.model_copy(update={"features": bad_features})
        match_repo = _FakeMatchRepo(finals=[_final(10, tournament_id=100, match_date=date(2026, 1, 5))])
        # §M12 is a construction-time invariant (Codex Fix A): a misconfigured
        # agent can't even be built, so it is never wired into a run — no row to
        # strand mid-run. The match repo is never touched.
        with pytest.raises(FeatureContractError, match="windows_days"):
            _agent(bad, mode="training", match_repo=match_repo)
        assert match_repo.for_training_calls == 0


# ---------------------------------------------------------------------------
# Scope + Elo handling
# ---------------------------------------------------------------------------
class TestTrainingScope:
    def test_runs_walk_writes_rows_no_prediction_calls(self, config: AppConfig) -> None:
        finals = [
            _final(10, tournament_id=100, match_date=date(2026, 1, 5)),
            _final(11, tournament_id=101, match_date=date(2026, 1, 6)),
        ]
        match_repo = _FakeMatchRepo(finals=finals)
        elo = _FakeEloRepo()
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(config, mode="training", match_repo=match_repo,
                       elo_repo=elo, feature_matrix_repo=fm)
        ctx, _ = _ctx(config)

        result = agent.run(ctx)

        assert result.ok is True
        assert match_repo.for_training_calls == 1
        assert match_repo.for_prediction_calls == 0      # never in training
        assert elo.career_counts_calls == 0              # walk, not reconstruction
        assert elo.inserts, "EloWalk must populate the ladder in training"
        assert len(fm.rows) == 2
        assert result.metrics["mode"] == "training"
        assert result.metrics["elo"]["fragments"] == 2
        assert result.metrics["scope"]["matches_written"] == 2


class TestPredictionScope:
    def test_reconstructs_counts_reads_ladder_no_walk(self, config: AppConfig) -> None:
        finals = [_final(10, tournament_id=100, match_date=date(2026, 1, 5))]
        slate = [_scheduled(20, tournament_id=200, start_ts=_NOW + timedelta(days=2))]
        match_repo = _FakeMatchRepo(finals=finals, scheduled=slate)
        elo = _FakeEloRepo(counts={1: 5, 2: 12})
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(config, mode="prediction", match_repo=match_repo,
                       elo_repo=elo, feature_matrix_repo=fm)
        ctx, _ = _ctx(config)

        result = agent.run(ctx)

        assert result.ok is True
        assert match_repo.for_training_calls == 1        # history index from finals
        assert match_repo.for_prediction_calls == 1      # loop set = the slate
        assert elo.career_counts_calls == 1              # reconstruction
        assert elo.inserts == []                          # NO walk on the predict path
        assert len(fm.rows) == 1
        assert fm.rows[0].match_id == 20
        assert result.metrics["mode"] == "prediction"
        assert result.metrics["elo"]["career_counts_players"] == 2


# ---------------------------------------------------------------------------
# Per-match fault isolation
# ---------------------------------------------------------------------------
class TestFaultIsolation:
    def test_none_tournament_skipped_and_dead_lettered(self, config: AppConfig) -> None:
        # decision 2: None tournament → skip + dead-letter → partial.
        finals = [
            _final(10, tournament_id=100, match_date=date(2026, 1, 5)),
            _final(11, tournament_id=999, match_date=date(2026, 1, 6)),  # unresolved
        ]
        match_repo = _FakeMatchRepo(finals=finals)
        dl = _FakeDeadLetter()
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(config, mode="training", match_repo=match_repo,
                       tournament_repo=_FakeTournamentRepo(none_for={999}),
                       feature_matrix_repo=fm, dead_letter=dl)
        ctx, _ = _ctx(config)

        result = agent.run(ctx)

        assert result.ok is False                         # something was skipped
        assert len(fm.rows) == 1 and fm.rows[0].match_id == 10
        assert len(dl.rows) == 1
        assert dl.rows[0].error["reason"] == "tournament_unresolved"
        assert dl.rows[0].run_id == ctx.run_id
        assert result.metrics["scope"]["matches_dead_lettered"] == 1

    def test_extractor_exception_isolated_and_redacted(self, config: AppConfig) -> None:
        # A repo error mid-extraction dead-letters that match only (prediction
        # mode → no walk to disturb); the secret in the message is redacted.
        slate = [
            _scheduled(20, tournament_id=200, start_ts=_NOW + timedelta(days=2)),
            _scheduled(21, tournament_id=201, start_ts=_NOW + timedelta(days=3)),
        ]
        match_repo = _FakeMatchRepo(finals=[], scheduled=slate)
        dl = _FakeDeadLetter()
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(config, mode="prediction", match_repo=match_repo,
                       tournament_repo=_FakeTournamentRepo(raise_for={201}),
                       feature_matrix_repo=fm, dead_letter=dl)
        ctx, _ = _ctx(config)

        result = agent.run(ctx)

        assert result.ok is False
        assert len(fm.rows) == 1 and fm.rows[0].match_id == 20
        assert len(dl.rows) == 1
        assert dl.rows[0].error["reason"] == "extraction_error"
        # §L10: the cause is redacted, never a raw repr leaking the secret.
        assert "shhh" not in str(dl.rows[0].error.get("cause", ""))


# ---------------------------------------------------------------------------
# Validate-before-write (C10)
# ---------------------------------------------------------------------------
class TestValidatorGate:
    def test_invalid_matrix_writes_nothing_and_fails(self, config: AppConfig) -> None:
        # A stub `rankings` extractor emits NO keys → R1 violations → the whole
        # slate is rejected: zero writes, fatal "feature_matrix_invalid".
        finals = [_final(10, tournament_id=100, match_date=date(2026, 1, 5))]
        match_repo = _FakeMatchRepo(finals=finals)
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(
            config, mode="training", match_repo=match_repo, feature_matrix_repo=fm,
            extractor_factories=[("rankings", lambda d: _Stub("rankings", {}))],
        )
        ctx, _ = _ctx(config)

        result = agent.run(ctx)

        assert result.ok is False
        assert fm.rows == []                              # C10: zero writes
        assert len(result.errors) == 1
        assert result.errors[0].code == "feature_matrix_invalid"
        assert result.metrics["scope"]["matches_written"] == 0


# ---------------------------------------------------------------------------
# Seeding + merge + heartbeat
# ---------------------------------------------------------------------------
class TestSeedingMergeHeartbeat:
    def test_seeds_all_seven_families(self, config: AppConfig) -> None:
        match_repo = _FakeMatchRepo()  # empty scope still seeds at startup
        fs = _FakeFeatureSpecRepo()
        agent = _agent(config, mode="training", match_repo=match_repo,
                       feature_spec_repo=fs)
        ctx, _ = _ctx(config)
        agent.run(ctx)

        expected_keys = {
            row.feature_key
            for fam in ("elo", "rankings", "form", "h2h", "surface",
                        "serve_return", "conditions")
            for row in specs._REGISTRY[fam]
        }
        seeded_keys = {fk for (fk, _v) in fs.specs}
        assert expected_keys <= seeded_keys

    def test_all_families_merge_into_one_clean_row(self, config: AppConfig) -> None:
        finals = [_final(10, tournament_id=100, match_date=date(2026, 1, 5))]
        match_repo = _FakeMatchRepo(finals=finals)
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(config, mode="training", match_repo=match_repo,
                       feature_matrix_repo=fm)
        ctx, _ = _ctx(config)
        agent.run(ctx)

        assert len(fm.rows) == 1
        payload = fm.rows[0].payload
        # one representative key per family must be present in the merged row
        for key in ("p1_elo_pre", "p1_rank_stale", "p1_win_rate_7d",
                    "h2h_matches", "surface_transition_type",
                    "p1_ace_rate_365d", "indoor"):
            assert key in payload, f"missing {key}"
        # H1: clean values — base Elo carries the 1500 cold-start fallback.
        assert payload["p1_elo_pre"] == 1500.0
        assert fm.rows[0].feature_set == config.features.feature_set
        assert fm.rows[0].as_of_ts.tzinfo is not None

    def test_empty_scope_succeeds_with_no_writes(self, config: AppConfig) -> None:
        match_repo = _FakeMatchRepo()
        fm = _FakeFeatureMatrixRepo()
        agent = _agent(config, mode="prediction", match_repo=match_repo,
                       feature_matrix_repo=fm)
        ctx, beats = _ctx(config)
        result = agent.run(ctx)
        assert result.ok is True
        assert fm.rows == []
        assert beats, "heartbeat must fire at least once"

    def test_heartbeat_fires_during_run(self, config: AppConfig) -> None:
        finals = [_final(10, tournament_id=100, match_date=date(2026, 1, 5))]
        agent = _agent(config, mode="training", match_repo=_FakeMatchRepo(finals=finals))
        ctx, beats = _ctx(config)
        agent.run(ctx)
        assert len(beats) >= 1
