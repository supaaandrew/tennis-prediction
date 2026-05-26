"""ModelingAgent orchestration tests (M1a).

Real control flow with in-memory fakes (no Docker): assemble -> split -> CV ->
final train -> serialize -> register+activate. Uses tiny base-learner
hyperparams + a small balanced dataset so the suite stays fast. Regressions for
the locked decisions: precondition, ≤1-active flip (clarification #1), §M21f
walkover drop, insufficient-data -> failed/zero-writes, DB-error redaction (§L10).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from tennis.agents.modeling import ModelingAgent
from tennis.agents.research.specs import _REGISTRY
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig
from tennis.core.contracts import Agent, AgentContext
from tennis.core.errors import StorageError
from tennis.core.lineage import Precondition
from tennis.core.logging import get_logger
from tennis.models.feature_set import resolve_model_feature_set
from tennis.storage.postgres.rows import FeatureMatrixRow, MatchRow, ModelRegistryRow

_NOW = datetime(2026, 5, 26, 6, 30, tzinfo=UTC)
_ACTIVE_SPECS = [row for rows in _REGISTRY.values() for row in rows]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeMatchRepo:
    def __init__(self, finals, *, raise_storage=False):
        self._finals = list(finals)
        self._raise = raise_storage
        self.for_training_calls = 0

    def for_training(self, *, season_start, season_end):
        self.for_training_calls += 1
        if self._raise:
            raise StorageError("connect failed postgres://u:topsecret@db/x")
        return list(self._finals)


class _FakeFeatureMatrixRepo:
    def __init__(self, rows):
        self._by_id = {r.match_id: r for r in rows}

    def list_for_matches(self, *, match_ids, feature_set):
        return [self._by_id[m] for m in match_ids if m in self._by_id]


class _FakeFeatureSpecRepo:
    def __init__(self, specs=_ACTIVE_SPECS):
        self._specs = list(specs)

    def list_active(self, *, feature_set):
        return list(self._specs)


class _FakeModelRegistryRepo:
    """Emulates the migration-005 ≤1-active partial unique index."""

    def __init__(self):
        self.rows: dict[str, ModelRegistryRow] = {}
        self.insert_calls = 0
        self.activate_calls: list[str] = []

    def insert(self, row):
        self.insert_calls += 1
        self.rows[row.version] = row
        return row

    def activate(self, version):
        self.activate_calls.append(version)
        for v, r in list(self.rows.items()):
            self.rows[v] = dataclasses.replace(r, is_active=(v == version))

    def active(self):
        return next((r for r in self.rows.values() if r.is_active), None)

    def get(self, version):
        return self.rows.get(version)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _shrink(config: AppConfig, *, artifact_dir: str) -> AppConfig:
    md = config.modeling
    return config.model_copy(
        update={
            "modeling": md.model_copy(
                update={
                    "artifact_dir": artifact_dir,
                    "splits": md.splits.model_copy(
                        update={"min_train_seasons": 2, "n_folds": 2}
                    ),
                    "calibration": md.calibration.model_copy(update={"tail_days": 1}),
                    "base_learners": md.base_learners.model_copy(
                        update={
                            "xgb": md.base_learners.xgb.model_copy(
                                update={"n_estimators": 8, "early_stopping_rounds": 3,
                                        "max_depth": 2}
                            ),
                            "lgbm": md.base_learners.lgbm.model_copy(
                                update={"n_estimators": 8, "early_stopping_rounds": 3,
                                        "num_leaves": 5}
                            ),
                        }
                    ),
                }
            )
        }
    )


def _training_set(seasons=4, per_season=16, start=2016, *, all_p1=False):
    matches: list[MatchRow] = []
    frows: list[FeatureMatrixRow] = []
    i = 0
    trans = ["same", "clay->hard", "none"]
    for s in range(seasons):
        y = start + s
        for k in range(per_season):
            mid = 1000 + i
            p1win = True if all_p1 else (k % 2 == 0)
            matches.append(
                MatchRow(
                    match_id=mid, tournament_id=y * 1000, round="R16",
                    match_date=date(y, 6, 1 + k), p1_id=1 + i, p2_id=9000 + i,
                    status="final", source="t", source_uid=f"u{i}",
                    winner_id=(1 + i) if p1win else (9000 + i),
                )
            )
            frows.append(
                FeatureMatrixRow(
                    match_id=mid, feature_set="v1",
                    as_of_ts=datetime(y, 5, 1, tzinfo=UTC),
                    payload={
                        "elo_diff_blended": (2.0 if p1win else -2.0) + (k % 3) * 0.1,
                        "p1_elo_pre": 1500.0 + i,
                        "p1_elo_surface_pre": 1500.0 + i,
                        "p2_elo_pre": 1500.0,
                        "p2_elo_surface_pre": 1500.0,
                        "p1_elo_blended_pre": 1500.0 + i,
                        "p2_elo_blended_pre": 1500.0,
                        "p1_elo_reliability_low": False,
                        "p2_elo_reliability_low": False,
                        "surface_transition_type": trans[k % 3],
                    },
                )
            )
            i += 1
    return matches, frows


def _agent(config, match_repo, fm_repo, reg_repo, spec_repo=None):
    return ModelingAgent(
        config=config,
        match_repo=match_repo,
        feature_matrix_repo=fm_repo,
        feature_spec_repo=spec_repo or _FakeFeatureSpecRepo(),
        model_registry_repo=reg_repo,
    )


def _ctx(config, heartbeats):
    return AgentContext(
        run_id=uuid4(), as_of=_NOW, config=config, db=None,
        clock=FrozenClock(_NOW), logger=get_logger("test"),
        heartbeat=lambda: heartbeats.append(1),
    )


@pytest.fixture
def shrunk(base_config, tmp_path):
    return _shrink(base_config, artifact_dir=str(tmp_path / "models"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestContract:
    def test_protocol_conformance(self, shrunk):
        agent = _agent(shrunk, _FakeMatchRepo([]), _FakeFeatureMatrixRepo([]),
                       _FakeModelRegistryRepo())
        assert isinstance(agent, Agent)
        assert agent.name == "modeling"

    def test_precondition_is_research_succeeded(self, shrunk):
        agent = _agent(shrunk, _FakeMatchRepo([]), _FakeFeatureMatrixRepo([]),
                       _FakeModelRegistryRepo())
        assert agent.lineage.preconditions == (
            Precondition(previous_agent="research", required_status="succeeded"),
        )


class TestSuccess:
    def test_succeeds_inserts_and_activates(self, shrunk):
        matches, frows = _training_set()
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        hb: list[int] = []
        result = agent.run(_ctx(shrunk, hb))
        assert result.ok is True
        assert reg.insert_calls == 1
        assert len(reg.activate_calls) == 1
        assert reg.active() is not None and reg.active().is_active is True

    def test_one_active_flip_supersedes_prior(self, shrunk):
        matches, frows = _training_set()
        reg = _FakeModelRegistryRepo()
        # pre-existing active model from a prior run
        reg.rows["OLD"] = ModelRegistryRow(
            version="OLD", trained_at=_NOW, feature_set="v1", algo="prev",
            hyperparams={}, metrics={}, artifact_uri="x", feature_hash="h",
            data_window_start=date(2010, 1, 1), data_window_end=date(2011, 1, 1),
            is_active=True,
        )
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        agent.run(_ctx(shrunk, []))
        actives = [v for v, r in reg.rows.items() if r.is_active]
        assert len(actives) == 1
        assert "OLD" not in actives
        assert reg.rows["OLD"].is_active is False

    def test_feature_hash_persisted(self, shrunk):
        matches, frows = _training_set()
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        expected = resolve_model_feature_set(_ACTIVE_SPECS).feature_hash
        row = reg.rows[result.metrics["version"]]
        assert row.feature_hash == expected
        assert result.metrics["feature_hash"] == expected

    def test_algo_is_base_only(self, shrunk):
        matches, frows = _training_set()
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        assert reg.rows[result.metrics["version"]].algo == "xgb+lgbm_base"

    def test_heartbeat_fired(self, shrunk):
        matches, frows = _training_set()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows),
                       _FakeModelRegistryRepo())
        hb: list[int] = []
        agent.run(_ctx(shrunk, hb))
        assert len(hb) >= 1

    def test_metrics_shape(self, shrunk):
        matches, frows = _training_set()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows),
                       _FakeModelRegistryRepo())
        m = agent.run(_ctx(shrunk, [])).metrics
        assert {"version", "feature_hash", "rows", "folds", "cv", "data_window",
                "dropped"} <= set(m)


class TestLabelHygiene:
    def test_walkover_dropped_from_training(self, shrunk):
        matches, frows = _training_set()
        # add a walkover with a feature row — only the §M21f filter should drop it
        wo = MatchRow(
            match_id=9999, tournament_id=2016_000, round="R16",
            match_date=date(2016, 6, 2), p1_id=7, p2_id=8, status="final",
            source="t", source_uid="wo", winner_id=7, walkover=True,
        )
        matches.append(wo)
        frows.append(FeatureMatrixRow(
            match_id=9999, feature_set="v1", as_of_ts=datetime(2016, 5, 1, tzinfo=UTC),
            payload={"elo_diff_blended": 1.0},
        ))
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        assert result.metrics["dropped"]["walkover"] == 1

    def test_invalid_winner_dropped(self, shrunk):
        # winner_id belongs to neither player -> corrupted; dropped, not mislabeled.
        matches, frows = _training_set()
        bad = MatchRow(
            match_id=8888, tournament_id=2016_000, round="R16",
            match_date=date(2016, 6, 3), p1_id=11, p2_id=12, status="final",
            source="t", source_uid="bad", winner_id=999_999,  # neither p1 nor p2
        )
        matches.append(bad)
        frows.append(FeatureMatrixRow(
            match_id=8888, feature_set="v1", as_of_ts=datetime(2016, 5, 1, tzinfo=UTC),
            payload={"elo_diff_blended": 1.0},
        ))
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        assert result.metrics["dropped"]["invalid_winner"] == 1


class TestFailure:
    def test_insufficient_data_fails_with_zero_writes(self, shrunk):
        matches, frows = _training_set(seasons=1, per_season=10)  # < min_train_seasons
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "insufficient_training_data"
        assert reg.insert_calls == 0

    def test_empty_training_fails(self, shrunk):
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo([]), _FakeFeatureMatrixRepo([]), reg)
        result = agent.run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "insufficient_training_data"
        assert reg.insert_calls == 0

    def test_db_error_fails_and_redacts(self, shrunk):
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo([], raise_storage=True),
                       _FakeFeatureMatrixRepo([]), reg)
        result = agent.run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "modeling_db_error"
        assert reg.insert_calls == 0
        # §L10: the credential in the DB URL must be scrubbed from the cause.
        assert "topsecret" not in (result.errors[0].cause or "")

    def test_degenerate_cv_fails_with_zero_writes(self, shrunk):
        # all-p1-win -> every fold's train block is single-class -> CV skips all
        # folds -> n_oof=0 -> activation gate raises (Codex HIGH); zero writes.
        matches, frows = _training_set(all_p1=True)
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "insufficient_training_data"
        assert reg.insert_calls == 0
        assert len(reg.activate_calls) == 0
