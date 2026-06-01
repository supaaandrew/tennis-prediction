"""MonitorAgent orchestration tests (§Q).

Real control flow with in-memory fakes + FrozenClock (no network, no Docker).
Regressions for the locked decisions: preconditions == () (A13 / §Q7); the
realized-outcome join + pending-but-counted rows (§Q1); per-window ECE/PSI/ROI
written to AgentResult.metrics in the Q6 shape; zero-settled → None metrics +
partial (item 3 / §Q5); active-model_version scoping; rolling PSI reference with
NO prior window → None, not a crash (item 2 / §Q4); DB error → failed with a
redacted cause (§Q5 / §L10); naive as_of rejected.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tennis.agents.monitor import MonitorAgent
from tennis.agents.orchestrator.pipeline import _FATAL_CODES, DailyPipeline
from tennis.core.clock import FrozenClock
from tennis.core.contracts import Agent, AgentContext
from tennis.core.errors import StorageError
from tennis.core.logging import get_logger
from tennis.storage.postgres.rows import MatchRow, ModelRegistryRow, PredictionRow

_NOW = datetime(2026, 5, 26, 6, 30, tzinfo=UTC)
_VERSION = "2016-2019-20260526T0630Z"
_SECRET_DSN = "connect failed postgres://user:topsecret@db/x"

# anchor = midnight UTC of as_of.date() = 2026-05-26 00:00.
#   30d current   = [2026-04-26, 2026-05-26)   -> CURRENT_TS lives here
#   30d reference = [2026-03-27, 2026-04-26)   -> REF_TS lives here
_CURRENT_TS = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
_REF_TS = datetime(2026, 4, 10, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _model_row(*, version=_VERSION, feature_set="v1", active=True) -> ModelRegistryRow:
    return ModelRegistryRow(
        version=version,
        trained_at=_NOW,
        feature_set=feature_set,
        algo="xgb+lgbm_stack_isotonic",
        hyperparams={},
        metrics={},
        artifact_uri="file:///models/x.joblib",
        feature_hash="abc123",
        data_window_start=date(2016, 1, 1),
        data_window_end=date(2019, 12, 31),
        is_active=active,
    )


def _pred(
    match_id, *, model_version=_VERSION, predicted_at=_CURRENT_TS, prob=0.62,
    implied=0.55,
) -> PredictionRow:
    return PredictionRow(
        match_id=match_id,
        model_version=model_version,
        predicted_at=predicted_at,
        p1_prob_raw=prob,
        p1_prob_cal=prob,
        p1_implied_decision=implied,
    )


def _match(match_id, *, status="final", winner="p1") -> MatchRow:
    p1_id, p2_id = match_id * 10 + 1, match_id * 10 + 2
    winner_id = p1_id if winner == "p1" else p2_id if winner == "p2" else None
    return MatchRow(
        match_id=match_id,
        tournament_id=2026001,
        round="QF",
        match_date=date(2026, 5, 9),
        p1_id=p1_id,
        p2_id=p2_id,
        status=status,
        source="t",
        source_uid=f"u{match_id}",
        start_ts=_CURRENT_TS,
        winner_id=winner_id,
    )


def _single_window_config(base_config, days):
    """A copy of base_config with monitor.windows_days narrowed to one window —
    lets a test assert a fully-populated (succeeded) run without arranging
    reference data for all three default windows."""
    monitor = base_config.monitor.model_copy(update={"windows_days": (days,)})
    return base_config.model_copy(update={"monitor": monitor})


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeRegistry:
    def __init__(self, active_row):
        self._active = active_row

    def active(self):
        return self._active


class _FakePredictionRepo:
    def __init__(self, rows, *, raise_exc=None):
        self._rows = list(rows)
        self._raise = raise_exc
        self.calls = []

    def list_for_window(self, *, model_version, since, until):
        self.calls.append((model_version, since, until))
        if self._raise is not None:
            raise self._raise
        return [
            r
            for r in self._rows
            if r.model_version == model_version and since <= r.predicted_at < until
        ]


class _FakeMatchRepo:
    def __init__(self, matches, *, raise_exc=None):
        self._by_id = {mm.match_id: mm for mm in matches}
        self._raise = raise_exc
        self.bulk_calls = 0

    def get(self, match_id):
        return self._by_id.get(match_id)

    def list_for_ids(self, *, match_ids):
        if not match_ids:  # empty fast-path — no DB round-trip, not a bulk call
            return {}
        self.bulk_calls += 1
        if self._raise is not None:
            raise self._raise
        return {mid: self._by_id[mid] for mid in match_ids if mid in self._by_id}


def _make_agent(
    config, *, active=None, preds=(), matches=(), prediction_repo=None, match_repo=None
):
    return MonitorAgent(
        config=config,
        model_registry_repo=_FakeRegistry(active),
        prediction_repo=prediction_repo or _FakePredictionRepo(preds),
        match_repo=match_repo or _FakeMatchRepo(matches),
    )


def _ctx(config, *, as_of=_NOW, heartbeats=None):
    hb = heartbeats if heartbeats is not None else []
    return AgentContext(
        run_id=uuid4(),
        as_of=as_of,
        config=config,
        db=None,
        clock=FrozenClock(as_of),
        logger=get_logger("test"),
        heartbeat=lambda: hb.append(1),
    )


# ---------------------------------------------------------------------------
# Contract (A13 / §Q7)
# ---------------------------------------------------------------------------
class TestContract:
    def test_protocol_conformance(self, base_config):
        agent = _make_agent(base_config, active=_model_row())
        assert isinstance(agent, Agent)
        assert agent.name == "monitor"

    def test_preconditions_empty_locks_a13(self, base_config):
        """A13/§Q7: the Monitor runs post-briefing REGARDLESS of upstream status,
        so it MUST declare zero preconditions. A precondition slipping in here
        would block the Monitor whenever an upstream stage was partial/failed —
        exactly the observability we need most. Fail loudly if that happens."""
        agent = _make_agent(base_config, active=_model_row())
        assert agent.lineage.preconditions == ()


# ---------------------------------------------------------------------------
# Happy path + Q6 metrics shape
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_succeeded_full_window_metrics(self, base_config):
        cfg = _single_window_config(base_config, 30)
        preds = [
            _pred(1, predicted_at=_CURRENT_TS, prob=0.62),
            _pred(2, predicted_at=_CURRENT_TS, prob=0.40),
            _pred(9, predicted_at=_REF_TS, prob=0.5),  # reference-window row → PSI
        ]
        matches = [_match(1, winner="p1"), _match(2, winner="p2")]
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        result = agent.run(_ctx(cfg))

        assert result.ok is True
        assert result.errors == ()
        w = result.metrics["windows"]["30"]
        assert w["n_settled"] == 2 and w["n_pending"] == 0
        assert w["ece"] is not None and w["psi"] is not None and w["roi"] is not None
        assert result.metrics["model_version"] == _VERSION
        assert result.metrics["as_of"] == _NOW.isoformat()

    def test_all_default_windows_present(self, base_config):
        # CURRENT_TS sits inside all three default current windows (30/60/90).
        preds = [_pred(1, predicted_at=_CURRENT_TS)]
        matches = [_match(1, winner="p1")]
        agent = _make_agent(base_config, active=_model_row(), preds=preds, matches=matches)

        result = agent.run(_ctx(base_config))

        assert set(result.metrics["windows"]) == {"30", "60", "90"}
        for w in result.metrics["windows"].values():
            assert w["n_settled"] == 1

    def test_run_day_prediction_included_in_current_window(self, base_config):
        """H1 regression — a prediction written at `as_of` (the run day, ~06:30
        UTC) must fall INSIDE the current window. A start-of-day midnight anchor
        excluded it, leaving the most-recent window empty on the day it ran."""
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(1, predicted_at=_NOW)]  # same day/instant as the monitor run
        matches = [_match(1, winner="p1")]
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        w = agent.run(_ctx(cfg)).metrics["windows"]["30"]
        assert w["n_settled"] == 1  # included, not excluded by a midnight anchor

    def test_pending_excluded_from_metrics_but_counted(self, base_config):
        cfg = _single_window_config(base_config, 30)
        preds = [
            _pred(1, predicted_at=_CURRENT_TS),  # final → settled
            _pred(2, predicted_at=_CURRENT_TS),  # scheduled → pending
        ]
        matches = [_match(1, winner="p1"), _match(2, status="scheduled", winner=None)]
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        w = agent.run(_ctx(cfg)).metrics["windows"]["30"]

        assert w["n_settled"] == 1
        assert w["n_pending"] == 1

    def test_final_without_winner_is_pending(self, base_config):
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(1, predicted_at=_CURRENT_TS)]
        # status final but winner_id None (data gap) → pending, never guessed.
        matches = [_match(1, status="final", winner=None)]
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        w = agent.run(_ctx(cfg)).metrics["windows"]["30"]
        assert w["n_settled"] == 0 and w["n_pending"] == 1

    def test_winner_not_in_match_is_pending(self, base_config):
        """§Q1 — winner_id ∉ {p1_id, p2_id} (data corruption) → pending (None
        label), NOT silently scored as a p2 win (0)."""
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(1, predicted_at=_CURRENT_TS)]
        bad = replace(_match(1, status="final"), winner_id=999_999)  # neither side
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=[bad])

        w = agent.run(_ctx(cfg)).metrics["windows"]["30"]
        assert w["n_settled"] == 0 and w["n_pending"] == 1

    def test_one_bulk_match_read_per_window(self, base_config):
        """F2 — the outcome join is ONE bulk read per window, not one get() per
        prediction (retires the N+1; §M18 discipline)."""
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(i, predicted_at=_CURRENT_TS) for i in range(1, 21)]
        matches = [_match(i, winner="p1") for i in range(1, 21)]
        match_repo = _FakeMatchRepo(matches)
        agent = _make_agent(
            cfg, active=_model_row(), preds=preds, match_repo=match_repo
        )

        result = agent.run(_ctx(cfg))
        assert result.metrics["windows"]["30"]["n_settled"] == 20
        assert match_repo.bulk_calls == 1  # one bulk read for 20 predictions


# ---------------------------------------------------------------------------
# Partial / no-data paths (item 3 / §Q5)
# ---------------------------------------------------------------------------
class TestPartialPaths:
    def test_zero_settled_window_none_metrics_partial(self, base_config):
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(1, predicted_at=_CURRENT_TS)]
        matches = [_match(1, status="scheduled", winner=None)]  # not final → pending
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        result = agent.run(_ctx(cfg))

        assert result.ok is False
        codes = {e.code for e in result.errors}
        assert codes == {"monitor_partial"}
        assert not (codes & _FATAL_CODES)  # partial, NOT failed
        w = result.metrics["windows"]["30"]
        assert w["ece"] is None and w["roi"] is None
        assert w["n_settled"] == 0

    def test_no_prior_window_psi_none_not_crash(self, base_config):
        """Item 2 / §Q4 — first-ever run: the reference window has no rows, so
        PSI must be None (not a crash), even with settled current-window data."""
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(1, predicted_at=_CURRENT_TS)]  # no _REF_TS row
        matches = [_match(1, winner="p1")]
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        result = agent.run(_ctx(cfg))

        w = result.metrics["windows"]["30"]
        assert w["psi"] is None
        assert w["ece"] is not None and w["roi"] is not None  # these still compute
        assert result.ok is False  # a None metric → partial
        assert {e.code for e in result.errors} == {"monitor_partial"}

    def test_no_active_model_partial_not_fatal(self, base_config):
        agent = _make_agent(base_config, active=None)

        result = agent.run(_ctx(base_config))

        assert result.ok is False
        codes = {e.code for e in result.errors}
        assert codes == {"monitor_no_active_model"}
        assert not (codes & _FATAL_CODES)
        assert result.metrics["windows"] == {}
        assert result.metrics["model_version"] is None


# ---------------------------------------------------------------------------
# Scoping + failure modes
# ---------------------------------------------------------------------------
class TestScopingAndFailures:
    def test_scoped_to_active_model_version(self, base_config):
        cfg = _single_window_config(base_config, 30)
        preds = [
            _pred(1, predicted_at=_CURRENT_TS),
            _pred(2, model_version="other-model", predicted_at=_CURRENT_TS),
        ]
        matches = [_match(1, winner="p1"), _match(2, winner="p1")]
        agent = _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)

        w = agent.run(_ctx(cfg)).metrics["windows"]["30"]
        assert w["n_settled"] == 1  # the "other-model" row is ignored

    def test_db_error_fails_with_redacted_cause(self, base_config):
        repo = _FakePredictionRepo([], raise_exc=StorageError(_SECRET_DSN))
        agent = _make_agent(base_config, active=_model_row(), prediction_repo=repo)

        result = agent.run(_ctx(base_config))

        assert result.ok is False
        assert len(result.errors) == 1
        err = result.errors[0]
        assert err.code == "monitor_db_error"
        assert err.code in _FATAL_CODES
        assert "topsecret" not in (err.cause or "")  # §L10 redaction

    def test_db_error_returns_q6_envelope(self, base_config):
        """F1 — even the fatal DB path returns the §Q6 envelope shape, so a
        downstream metrics reader sees the same fixed keys it always expects."""
        repo = _FakePredictionRepo([], raise_exc=StorageError(_SECRET_DSN))
        agent = _make_agent(base_config, active=_model_row(), prediction_repo=repo)

        result = agent.run(_ctx(base_config))

        assert result.errors[0].code == "monitor_db_error"
        assert set(result.metrics) == {"model_version", "as_of", "windows"}
        assert result.metrics["model_version"] is None
        assert result.metrics["windows"] == {}

    def test_match_repo_db_error_fails(self, base_config):
        """A StorageError from the bulk match read (mid-window) propagates to
        failed with a redacted cause — not a half-written metrics dict."""
        cfg = _single_window_config(base_config, 30)
        preds = [_pred(1, predicted_at=_CURRENT_TS)]
        match_repo = _FakeMatchRepo([], raise_exc=StorageError(_SECRET_DSN))
        agent = _make_agent(
            cfg, active=_model_row(), preds=preds, match_repo=match_repo
        )

        result = agent.run(_ctx(cfg))

        assert result.ok is False
        assert result.errors[0].code == "monitor_db_error"
        assert "topsecret" not in (result.errors[0].cause or "")

    def test_non_utc_as_of_normalized_to_utc(self, base_config):
        """F3 — window math keys off the UTC calendar day. A non-UTC as_of at
        the SAME instant must yield identical windows; its local date must not
        shift the anchor. The 27th-UTC row must be EXCLUDED by both runs — a
        midnight-LOCAL anchor on the +5 run would have wrongly included it."""
        cfg = _single_window_config(base_config, 30)
        instant = datetime(2026, 5, 26, 23, 30, tzinfo=UTC)
        as_of_plus5 = instant.astimezone(timezone(timedelta(hours=5)))  # local 27th
        preds = [
            _pred(1, predicted_at=_CURRENT_TS),
            _pred(2, predicted_at=datetime(2026, 5, 27, 12, 0, tzinfo=UTC)),
        ]
        matches = [_match(1, winner="p1"), _match(2, winner="p1")]

        w_utc = (
            _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)
            .run(_ctx(cfg, as_of=instant))
            .metrics["windows"]["30"]
        )
        w_p5 = (
            _make_agent(cfg, active=_model_row(), preds=preds, matches=matches)
            .run(_ctx(cfg, as_of=as_of_plus5))
            .metrics["windows"]["30"]
        )

        assert w_utc == w_p5
        assert w_utc["n_settled"] == 1  # only _CURRENT_TS; the 27th-UTC row excluded

    def test_naive_as_of_rejected(self, base_config):
        agent = _make_agent(base_config, active=_model_row())
        # Bypass AgentContext (which also rejects naive) to prove the agent's own
        # guard fires — a ValueError, never a silent mis-fired PIT window.
        stub = SimpleNamespace(as_of=datetime(2026, 5, 26, 6, 30), heartbeat=lambda: None)
        with pytest.raises(ValueError):
            agent.run(stub)

    def test_status_mapping(self, base_config):
        """The orchestrator's `_map_status` maps a clean Monitor result to
        'succeeded', a degraded one to 'partial', and a DB error to 'failed'
        (mirrors B1's `_map_status` regression)."""
        cfg = _single_window_config(base_config, 30)
        ok_agent = _make_agent(
            cfg,
            active=_model_row(),
            preds=[_pred(1, predicted_at=_CURRENT_TS), _pred(9, predicted_at=_REF_TS)],
            matches=[_match(1, winner="p1")],
        )
        assert DailyPipeline._map_status(ok_agent.run(_ctx(cfg))) == "succeeded"

        partial_agent = _make_agent(
            cfg, active=_model_row(),
            preds=[_pred(1, predicted_at=_CURRENT_TS)],  # no reference → PSI None
            matches=[_match(1, winner="p1")],
        )
        assert DailyPipeline._map_status(partial_agent.run(_ctx(cfg))) == "partial"

        db_repo = _FakePredictionRepo([], raise_exc=StorageError(_SECRET_DSN))
        fail_agent = _make_agent(cfg, active=_model_row(), prediction_repo=db_repo)
        assert DailyPipeline._map_status(fail_agent.run(_ctx(cfg))) == "failed"


# ---------------------------------------------------------------------------
# §T6 — matchstat quota-burn metric in the §Q6 envelope
# ---------------------------------------------------------------------------
class _FakeMatchstatClient:
    """Records property reads; programmable to return values OR raise."""

    def __init__(
        self,
        *,
        quota_remaining: int = 481,
        monthly_quota: int = 500,
        raise_on_read: bool = False,
    ) -> None:
        self._qr = quota_remaining
        self._mq = monthly_quota
        self._raise = raise_on_read
        self.read_count = 0

    @property
    def quota_remaining(self) -> int:
        self.read_count += 1
        if self._raise:
            raise RuntimeError("simulated quota-read failure")
        return self._qr

    @property
    def monthly_quota(self) -> int:
        return self._mq


def _make_agent_with_ms(
    config, *, active=None, preds=(), matches=(),
    matchstat_client=None,
):
    return MonitorAgent(
        config=config,
        model_registry_repo=_FakeRegistry(active),
        prediction_repo=_FakePredictionRepo(preds),
        match_repo=_FakeMatchRepo(matches),
        matchstat_client=matchstat_client,
    )


class TestT6MatchstatEnvelope:
    def test_envelope_carries_quota_remaining_when_client_set(self, base_config):
        cfg = _single_window_config(base_config, 30)
        ms = _FakeMatchstatClient(quota_remaining=481, monthly_quota=500)
        agent = _make_agent_with_ms(
            cfg, active=_model_row(),
            preds=[_pred(1, predicted_at=_CURRENT_TS)],
            matches=[_match(1, winner="p1")],
            matchstat_client=ms,
        )
        result = agent.run(_ctx(cfg))
        assert "matchstat" in result.metrics
        assert result.metrics["matchstat"] == {
            "quota_remaining": 481,
            "monthly_quota": 500,
        }
        assert ms.read_count >= 1

    def test_envelope_omits_matchstat_when_client_is_none(self, base_config):
        cfg = _single_window_config(base_config, 30)
        agent = _make_agent_with_ms(
            cfg, active=_model_row(),
            preds=[_pred(1, predicted_at=_CURRENT_TS)],
            matches=[_match(1, winner="p1")],
            matchstat_client=None,
        )
        result = agent.run(_ctx(cfg))
        assert "matchstat" not in result.metrics
        # §Q6 envelope shape unchanged (model_version, as_of, windows).
        assert set(result.metrics) == {"model_version", "as_of", "windows"}

    def test_matchstat_read_failure_logs_warn_and_envelope_unchanged(
        self, base_config
    ) -> None:
        """A failing matchstat property read must NEVER block the monitor
        write — best-effort: warn + omit the `matchstat` key entirely so
        the §Q6 envelope shape stays the same."""
        cfg = _single_window_config(base_config, 30)
        ms = _FakeMatchstatClient(raise_on_read=True)
        agent = _make_agent_with_ms(
            cfg, active=_model_row(),
            preds=[_pred(1, predicted_at=_CURRENT_TS)],
            matches=[_match(1, winner="p1")],
            matchstat_client=ms,
        )
        result = agent.run(_ctx(cfg))
        assert "matchstat" not in result.metrics
        # Run still succeeded / partial on its own merits — never failed by the
        # matchstat read.
        assert result.errors == () or {e.code for e in result.errors} == {
            "monitor_partial"
        }
