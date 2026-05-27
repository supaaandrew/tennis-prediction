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
    def __init__(self, finals, *, scheduled=(), raise_storage=False):
        self._finals = list(finals)
        self._scheduled = list(scheduled)
        self._raise = raise_storage
        self.for_training_calls = 0

    def for_training(self, *, season_start, season_end):
        self.for_training_calls += 1
        if self._raise:
            raise StorageError("connect failed postgres://u:topsecret@db/x")
        return list(self._finals)

    def for_prediction(self, *, as_of, lookforward_days):
        if self._raise:
            raise StorageError("connect failed postgres://u:topsecret@db/x")
        return list(self._scheduled)


class _FakePredictionRepo:
    def __init__(self, *, raise_for=()):
        self.rows: dict[tuple[int, str], object] = {}
        self.upsert_calls = 0
        self._raise_for = set(raise_for)

    def upsert(self, row):
        self.upsert_calls += 1
        if row.match_id in self._raise_for:
            raise StorageError("upsert failed postgres://u:topsecret@db/x")
        self.rows[(row.match_id, row.model_version)] = row
        return row

    def get(self, *, match_id, model_version):
        return self.rows.get((match_id, model_version))

    def list_for_window(self, *, model_version, since, until):
        return [r for (_, v), r in self.rows.items() if v == model_version]


class _FakeDeadLetterRepo:
    def __init__(self):
        self.rows: list[object] = []

    def append(self, row):
        self.rows.append(row)


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
                    # tail_days ~200 carves the WHOLE last season as a
                    # non-straddling calibration tail (one tournament/season in
                    # the fixture), and a low min_calibration_samples lets that
                    # 16-row tail calibrate → non-degraded success path.
                    "calibration": md.calibration.model_copy(
                        update={"tail_days": 200, "min_calibration_samples": 8}
                    ),
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

    def test_algo_is_stacked(self, shrunk):
        # M1b supersedes the M1a base-only tag with the stacked+calibrated algo.
        matches, frows = _training_set()
        reg = _FakeModelRegistryRepo()
        agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(shrunk, []))
        assert reg.rows[result.metrics["version"]].algo == "xgb+lgbm_stack_platt"

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
        assert {"version", "feature_hash", "rows", "folds", "data_window", "dropped",
                "logloss", "brier", "n_oof", "ece_oof", "roi_kelly", "roi_kelly_shin",
                "roi_kelly_proportional", "tail_logloss", "tail_brier", "tail_ece",
                "calibration_degraded"} <= set(m)


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


class TestCalibrationDegraded:
    def test_empty_tail_returns_partial_but_activates(self, shrunk):
        # tail_days=1 → the last season's tournament straddles the train/tail seam
        # and is dropped from both → EMPTY tail → degraded passthrough → partial
        # (pre-step 5.1). The model is STILL registered + active (served degraded).
        cfg = shrunk.model_copy(
            update={
                "modeling": shrunk.modeling.model_copy(
                    update={
                        "calibration": shrunk.modeling.calibration.model_copy(
                            update={"tail_days": 1}
                        )
                    }
                )
            }
        )
        matches, frows = _training_set()
        reg = _FakeModelRegistryRepo()
        agent = _agent(cfg, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
        result = agent.run(_ctx(cfg, []))
        assert result.ok is False                       # → 'partial'
        assert result.errors[0].code == "calibration_degraded"
        assert result.metrics["calibration_degraded"] is True
        assert reg.insert_calls == 1                     # still registered
        assert reg.active() is not None                  # still served


# ---------------------------------------------------------------------------
# Prediction mode (M1b)
# ---------------------------------------------------------------------------
def _scheduled_set(n=4, *, with_odds=True):
    matches: list[MatchRow] = []
    frows: list[FeatureMatrixRow] = []
    for k in range(n):
        mid = 5000 + k
        matches.append(
            MatchRow(
                match_id=mid, tournament_id=2027_000, round="R16",
                match_date=date(2027, 6, 1 + k), p1_id=20 + k, p2_id=9020 + k,
                status="scheduled", source="t", source_uid=f"s{k}", winner_id=None,
                start_ts=datetime(2027, 6, 1 + k, 12, tzinfo=UTC),
            )
        )
        payload = {
            "elo_diff_blended": 0.8, "p1_elo_pre": 1600.0, "p1_elo_surface_pre": 1600.0,
            "p2_elo_pre": 1500.0, "p2_elo_surface_pre": 1500.0,
            "p1_elo_blended_pre": 1600.0, "p2_elo_blended_pre": 1500.0,
            "p1_elo_reliability_low": False, "p2_elo_reliability_low": False,
            "surface_transition_type": "same",
        }
        if with_odds:
            payload["p1_implied_pinnacle_decision"] = 0.55       # Shin decision
            payload["p1_implied_proportional_decision"] = 0.56
            payload["p1_implied_pinnacle_opening"] = 0.54
            payload["p1_implied_pinnacle_closing"] = 0.99        # must be ignored
            payload["odds_drift_to_close"] = 0.42                # must be ignored
        frows.append(
            FeatureMatrixRow(
                match_id=mid, feature_set="v1",
                as_of_ts=datetime(2027, 5, 1, tzinfo=UTC), payload=payload,
            )
        )
    return matches, frows


def _train_active(shrunk) -> _FakeModelRegistryRepo:
    """Train a real model so prediction mode has an active artifact to load."""
    matches, frows = _training_set()
    reg = _FakeModelRegistryRepo()
    agent = _agent(shrunk, _FakeMatchRepo(matches), _FakeFeatureMatrixRepo(frows), reg)
    assert agent.run(_ctx(shrunk, [])).ok is True
    return reg


def _pred_agent(shrunk, reg, sched_matches, sched_frows, pred_repo, dl_repo):
    return ModelingAgent(
        mode="prediction", config=shrunk,
        match_repo=_FakeMatchRepo([], scheduled=sched_matches),
        feature_matrix_repo=_FakeFeatureMatrixRepo(sched_frows),
        feature_spec_repo=_FakeFeatureSpecRepo(), model_registry_repo=reg,
        prediction_repo=pred_repo, dead_letter_repo=dl_repo,
    )


class TestModeConstruction:
    def test_unknown_mode_raises(self, shrunk):
        with pytest.raises(ValueError, match="unknown mode"):
            ModelingAgent(
                mode="bogus", config=shrunk, match_repo=_FakeMatchRepo([]),
                feature_matrix_repo=_FakeFeatureMatrixRepo([]),
                feature_spec_repo=_FakeFeatureSpecRepo(),
                model_registry_repo=_FakeModelRegistryRepo(),
            )

    def test_prediction_mode_requires_repos(self, shrunk):
        with pytest.raises(ValueError, match="requires prediction_repo"):
            ModelingAgent(
                mode="prediction", config=shrunk, match_repo=_FakeMatchRepo([]),
                feature_matrix_repo=_FakeFeatureMatrixRepo([]),
                feature_spec_repo=_FakeFeatureSpecRepo(),
                model_registry_repo=_FakeModelRegistryRepo(),
            )


class TestPrediction:
    def test_no_active_model_returns_failed(self, shrunk):
        # prediction mode + empty registry → failed, ZERO writes, no exception.
        reg = _FakeModelRegistryRepo()  # empty
        sched, sfrows = _scheduled_set()
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        agent = _pred_agent(shrunk, reg, sched, sfrows, pred, dl)
        result = agent.run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "no_active_model"
        assert pred.upsert_calls == 0

    def test_base_only_model_not_servable_returns_failed(self, shrunk, tmp_path):
        # An M1a base-only artifact (stacker/calibrator None) cannot be scored by
        # M1b prediction → failed, zero writes (no crash on a None stacker).
        from tennis.models.artifacts import TrainedModel, save_artifact
        from tennis.models.base_learners import TrainedBaseLearners
        from tennis.models.feature_set import resolve_model_feature_set

        fs = resolve_model_feature_set(_ACTIVE_SPECS)
        base_only = TrainedModel(
            base_learners=TrainedBaseLearners(xgb={"x": 1}, lgbm={"l": 2}),
            feature_set=fs, algo="xgb+lgbm_base",  # no stacker/calibrator
        )
        uri = save_artifact(base_only, artifact_dir=str(tmp_path / "m"), version="OLD")
        reg = _FakeModelRegistryRepo()
        reg.rows["OLD"] = ModelRegistryRow(
            version="OLD", trained_at=_NOW, feature_set="v1", algo="xgb+lgbm_base",
            hyperparams={}, metrics={}, artifact_uri=uri, feature_hash=fs.feature_hash,
            data_window_start=date(2018, 1, 1), data_window_end=date(2019, 1, 1),
            is_active=True,
        )
        sched, sfrows = _scheduled_set(n=2)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "no_active_model"
        assert pred.upsert_calls == 0

    def test_feature_set_mismatch_returns_failed(self, shrunk):
        # Codex M1b HIGH: the active model's family must match config; a drift
        # fails fast (zero writes) rather than scoring the wrong payloads.
        reg = _train_active(shrunk)
        v = next(iter(reg.rows))
        reg.rows[v] = dataclasses.replace(reg.rows[v], feature_set="v_old")
        sched, sfrows = _scheduled_set(n=2)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert result.ok is False
        assert result.errors[0].code == "feature_set_mismatch"
        assert pred.upsert_calls == 0

    def test_one_bad_row_isolated_others_written(self, shrunk):
        # Codex M1b HIGH: a row-level scoring fault must not abort the slate —
        # it is dead-lettered while the other rows still upsert (per-match §L2).
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=3)
        sfrows[1].payload["p1_implied_pinnacle_decision"] = "not_a_number"
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert len(pred.rows) == 2          # the two good rows still written
        assert len(dl.rows) == 1            # the bad row isolated + dead-lettered
        assert result.ok is False           # → partial

    def test_scores_and_writes_predictions(self, shrunk):
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=3)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert result.ok is True
        assert pred.upsert_calls == 3
        assert result.metrics["n_predictions"] == 3
        # probabilities are valid + in range.
        for row in pred.rows.values():
            assert 0.0 <= row.p1_prob_cal <= 1.0

    def test_live_rows_closing_fields_always_null(self, shrunk):
        # §M19/§15.4: even though the payload carries closing+drift values, every
        # written live row NULLs them globally.
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=3, with_odds=True)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert pred.rows
        for row in pred.rows.values():
            assert row.p1_implied_close is None
            assert row.odds_drift_to_close is None

    def test_missing_odds_edge_null_but_written(self, shrunk):
        # C9: no market keys → edge_* NULL, prediction STILL written.
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=2, with_odds=False)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert result.ok is True
        assert pred.upsert_calls == 2
        for row in pred.rows.values():
            assert row.edge_p1_shin is None
            assert row.kelly_fraction_p1 is None  # no usable odds → NULL

    def test_no_feature_row_dead_lettered(self, shrunk):
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=3)
        sfrows = sfrows[:2]  # drop the 3rd match's feature row
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert pred.upsert_calls == 2
        assert len(dl.rows) == 1
        assert result.ok is False  # → partial (a match was dead-lettered)

    def test_upsert_storage_error_dead_lettered(self, shrunk):
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=3)
        pred = _FakePredictionRepo(raise_for={5001})  # one row fails to upsert
        dl = _FakeDeadLetterRepo()
        result = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert len(pred.rows) == 2          # 2 succeeded
        assert len(dl.rows) == 1            # 1 dead-lettered
        assert result.ok is False           # → partial
        # §L10: credential scrubbed from the dead-letter cause.
        assert "topsecret" not in str(dl.rows[0].error)

    def test_upsert_idempotent(self, shrunk):
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=2)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        agent = _pred_agent(shrunk, reg, sched, sfrows, pred, dl)
        agent.run(_ctx(shrunk, []))
        agent.run(_ctx(shrunk, []))  # re-run same slate
        assert len(pred.rows) == 2  # keyed on (match_id, model_version) — no dupes

    def test_metrics_omit_roi_kelly_surface_tail(self, shrunk):
        reg = _train_active(shrunk)
        sched, sfrows = _scheduled_set(n=2)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        m = _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, [])).metrics
        assert "roi_kelly" not in m                       # training-only (pre-step 6.2)
        assert {"tail_logloss", "tail_brier", "tail_ece", "n_predictions",
                "n_bets", "same_day_kelly_exposure"} <= set(m)

    def test_noise_not_applied_in_prediction_mode(self, shrunk, monkeypatch):
        # H1 / pre-step 5.5: noise is training-only; the predict path must never
        # touch the feature matrix with noise.
        import tennis.agents.modeling.agent as agent_mod

        reg = _train_active(shrunk)
        calls: list[int] = []
        monkeypatch.setattr(
            agent_mod, "apply_noise", lambda X, config: calls.append(1) or X
        )
        sched, sfrows = _scheduled_set(n=2)
        pred, dl = _FakePredictionRepo(), _FakeDeadLetterRepo()
        _pred_agent(shrunk, reg, sched, sfrows, pred, dl).run(_ctx(shrunk, []))
        assert calls == []  # apply_noise never invoked on the prediction path
