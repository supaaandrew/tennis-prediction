"""BriefingAgent orchestration tests (B1).

Real control flow with in-memory fakes + mock LLM/SMTP clients (no network, no
Docker). Regressions for the locked decisions: precondition (`modeling`
succeeded); no active model → failed/zero-email; no qualifying edges → failed
(incl. the NULL-only-slate case, §N4); succeeded; partial when a surfaced row is
no-market (C9); SMTP failure → failed + redacted cause (§L10); LLM failure
isolated → email still sent (§N4); filters to the active model_version; §N3
slate window; per-match context error dead-lettered (§L2); top-level DB error.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from tennis.agents.briefing import BriefingAgent
from tennis.agents.briefing.agent import _classify
from tennis.agents.orchestrator.pipeline import _FATAL_CODES, DailyPipeline
from tennis.core.clock import FrozenClock
from tennis.core.contracts import Agent, AgentContext, AgentError, AgentResult
from tennis.core.errors import BriefingEmailError, BriefingLlmError, StorageError
from tennis.core.lineage import Precondition
from tennis.core.logging import get_logger
from tennis.storage.postgres.rows import (
    BriefingDeliveryRow,
    FeatureMatrixRow,
    MatchRow,
    ModelRegistryRow,
    PlayerRow,
    PredictionRow,
    TournamentRow,
)

_NOW = datetime(2026, 5, 26, 6, 30, tzinfo=UTC)
_VERSION = "2016-2019-20260526T0630Z"
_SECRET_DSN = "connect failed postgres://user:topsecret@db/x"
_SECRET_TOKEN = "smtp auth failed token=SUPERSECRET"


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
        metrics={"tail_ece": 0.03},
        artifact_uri="file:///models/x.joblib",
        feature_hash="abc123",
        data_window_start=date(2016, 1, 1),
        data_window_end=date(2019, 12, 31),
        is_active=active,
    )


def _pred(
    match_id,
    *,
    model_version=_VERSION,
    edge1=None,
    edge2=None,
    prop1=None,
    prop2=None,
    k1=None,
    k2=None,
    prob=0.62,
) -> PredictionRow:
    return PredictionRow(
        match_id=match_id,
        model_version=model_version,
        predicted_at=_NOW,
        p1_prob_raw=prob,
        p1_prob_cal=prob,
        p1_implied_decision=0.55,
        edge_p1_shin=edge1,
        edge_p2_shin=edge2,
        edge_p1_proportional=prop1,
        edge_p2_proportional=prop2,
        kelly_fraction_p1=k1,
        kelly_fraction_p2=k2,
    )


def _qualifying(match_id, **kw):
    """A row with a real Shin edge above the default min_edge_to_log (0.01)."""
    return _pred(match_id, edge1=0.05, edge2=-0.04, k1=0.02, k2=0.0, **kw)


def _below_threshold(match_id, **kw):
    return _pred(match_id, edge1=0.005, edge2=-0.005, k1=0.0, k2=0.0, **kw)


def _no_market(match_id, **kw):
    """C9 — no odds existed: every edge + Kelly NULL."""
    return _pred(match_id, **kw)


def _match(match_id, *, p1_id=None, p2_id=None, tournament_id=2026001) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        tournament_id=tournament_id,
        round="QF",
        match_date=date(2026, 5, 26),
        p1_id=p1_id if p1_id is not None else match_id * 10 + 1,
        p2_id=p2_id if p2_id is not None else match_id * 10 + 2,
        status="scheduled",
        source="t",
        source_uid=f"u{match_id}",
        start_ts=_NOW,
    )


def _player(player_id, name) -> PlayerRow:
    return PlayerRow(player_id=player_id, full_name=name, source="t", source_uid=f"p{player_id}")


def _tourn(tournament_id=2026001, *, name="Test Open", surface="Hard") -> TournamentRow:
    return TournamentRow(
        tournament_id=tournament_id,
        season=2026,
        slug="test-open",
        name=name,
        tier="ATP250",
        surface=surface,
        indoor=False,
    )


def _fmrow(match_id) -> FeatureMatrixRow:
    return FeatureMatrixRow(
        match_id=match_id,
        feature_set="v1",
        as_of_ts=_NOW,
        payload={"elo_diff_blended": 1.5},
    )


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
        return [r for r in self._rows if r.model_version == model_version]


class _FakeMatchRepo:
    def __init__(self, matches, *, raise_for=()):
        self._by_id = {m.match_id: m for m in matches}
        self._raise_for = set(raise_for)

    def get(self, match_id):
        if match_id in self._raise_for:
            raise StorageError(_SECRET_DSN)
        return self._by_id.get(match_id)


class _FakePlayerRepo:
    def __init__(self, players):
        self._by_id = {p.player_id: p for p in players}

    def get(self, player_id):
        return self._by_id.get(player_id)


class _FakeTournamentRepo:
    def __init__(self, tourns):
        self._by_id = {t.tournament_id: t for t in tourns}

    def get(self, tournament_id):
        return self._by_id.get(tournament_id)


class _FakeFeatureMatrixRepo:
    def __init__(self, rows=()):
        self._by_id = {r.match_id: r for r in rows}

    def list_for_matches(self, *, match_ids, feature_set):
        return [self._by_id[m] for m in match_ids if m in self._by_id]


class _FakeDeadLetter:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


class _FakeLlm:
    def __init__(self, text="Narrative text.", raise_exc=None):
        self._text = text
        self._raise = raise_exc
        self.calls = 0

    def generate(self, *, prompt):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._text


class _FakeEmail:
    def __init__(self, raise_exc=None):
        self._raise = raise_exc
        self.sent = []

    def send(self, *, subject, body):
        if self._raise is not None:
            raise self._raise
        self.sent.append((subject, body))


class _FakeDelivery:
    """§N5/§S5 idempotency repo fake. `existing` is the row `get` returns (a
    prior delivery) or None; `get_raise`/`record_raise` inject failures."""

    def __init__(self, *, existing=None, get_raise=None, record_raise=None):
        self._existing = existing
        self._get_raise = get_raise
        self._record_raise = record_raise
        self.recorded = []
        self.get_calls = 0

    def get(self, *, briefing_day_utc, model_version):
        self.get_calls += 1
        if self._get_raise is not None:
            raise self._get_raise
        return self._existing

    def record(self, row):
        if self._record_raise is not None:
            raise self._record_raise
        self.recorded.append(row)


# ---------------------------------------------------------------------------
# Agent builder + ctx
# ---------------------------------------------------------------------------
def _make_agent(
    config,
    *,
    active=None,
    preds=(),
    matches=(),
    players=(),
    tourns=(),
    fmrows=(),
    llm=None,
    email=None,
    dead=None,
    delivery=None,
    match_repo=None,
    prediction_repo=None,
):
    return BriefingAgent(
        config=config,
        model_registry_repo=_FakeRegistry(active),
        prediction_repo=prediction_repo or _FakePredictionRepo(preds),
        match_repo=match_repo or _FakeMatchRepo(matches),
        player_repo=_FakePlayerRepo(players),
        tournament_repo=_FakeTournamentRepo(tourns),
        feature_matrix_repo=_FakeFeatureMatrixRepo(fmrows),
        dead_letter_repo=dead or _FakeDeadLetter(),
        briefing_delivery_repo=delivery or _FakeDelivery(),
        llm_client=llm or _FakeLlm(),
        email_client=email or _FakeEmail(),
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


def _full_slate(match_ids):
    """matches + players + one tournament covering the given ids."""
    matches = [_match(mid) for mid in match_ids]
    players = []
    for mid in match_ids:
        players.append(_player(mid * 10 + 1, f"P{mid}a"))
        players.append(_player(mid * 10 + 2, f"P{mid}b"))
    return matches, players, [_tourn()]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
class TestContract:
    def test_protocol_conformance(self, base_config):
        agent = _make_agent(base_config, active=_model_row())
        assert isinstance(agent, Agent)
        assert agent.name == "briefing"

    def test_precondition_is_modeling_succeeded(self, base_config):
        agent = _make_agent(base_config, active=_model_row())
        assert agent.lineage.preconditions == (
            Precondition(previous_agent="modeling", required_status="succeeded"),
        )


# ---------------------------------------------------------------------------
# Failed paths (zero email)
# ---------------------------------------------------------------------------
class TestFailedPaths:
    def test_no_active_model_fails_no_email(self, base_config):
        email = _FakeEmail()
        llm = _FakeLlm()
        agent = _make_agent(base_config, active=None, email=email, llm=llm)
        result = agent.run(_ctx(base_config))
        assert result.ok is False
        assert {e.code for e in result.errors} == {"no_active_model"}
        assert email.sent == []
        assert llm.calls == 0

    def test_no_qualifying_edges_fails_no_email(self, base_config):
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_below_threshold(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is False
        assert {e.code for e in result.errors} == {"no_qualifying_predictions"}
        assert email.sent == []

    def test_null_edge_only_slate_fails_no_email(self, base_config):
        """Locked §N4 regression: a no-market-only slate has zero qualifying
        rows → failed, ZERO email (NOT partial)."""
        matches, players, tourns = _full_slate([1, 2])
        email = _FakeEmail()
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_no_market(1), _no_market(2)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is False
        assert {e.code for e in result.errors} == {"no_qualifying_predictions"}
        assert email.sent == []

    def test_top_level_db_error_fails_no_email(self, base_config):
        email = _FakeEmail()
        repo = _FakePredictionRepo([], raise_exc=StorageError(_SECRET_DSN))
        agent = _make_agent(
            base_config, active=_model_row(), prediction_repo=repo, email=email
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is False
        codes = {e.code for e in result.errors}
        assert codes == {"briefing_db_error"}
        assert email.sent == []
        # §L10 — the DSN password is redacted in the stored cause.
        cause = result.errors[0].cause or ""
        assert "topsecret" not in cause
        assert "***" in cause


# ---------------------------------------------------------------------------
# Succeeded / partial
# ---------------------------------------------------------------------------
class TestSucceededPartial:
    def test_succeeded_sends_email(self, base_config):
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            fmrows=[_fmrow(1)],
            email=email,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is True
        assert result.errors == ()
        assert len(email.sent) == 1
        subject, body = email.sent[0]
        assert subject == "Tennis edges — 2026-05-26"
        assert "P1a" in body and "P1b" in body
        assert result.metrics["n_surfaced"] == 1
        assert result.metrics["narrative_degraded"] is False

    def test_partial_when_surfaced_row_has_null_edges(self, base_config):
        """C9 — a qualifying edge triggers the send; a co-surfaced no-market row
        is rendered (never dropped) and downgrades the run to partial."""
        matches, players, tourns = _full_slate([1, 2])
        email = _FakeEmail()
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1), _no_market(2)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is False  # → 'partial'
        assert {e.code for e in result.errors} == {"briefing_partial"}
        assert len(email.sent) == 1
        assert result.metrics["n_no_market"] == 1
        assert result.metrics["n_surfaced"] == 2
        _, body = email.sent[0]
        assert "no market" in body


# ---------------------------------------------------------------------------
# LLM / SMTP isolation
# ---------------------------------------------------------------------------
class TestClientFailures:
    def test_llm_failure_email_still_sent(self, base_config):
        """Named regression (§N4): LLM raises → narrative_degraded + email STILL
        sent → status decided by the edge rule (succeeded here). No raw secret
        leaks and the send happens exactly once."""
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        llm = _FakeLlm(raise_exc=BriefingLlmError("llm down apiKey=LEAKYKEY"))
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
            llm=llm,
        )
        result = agent.run(_ctx(base_config))
        assert llm.calls == 1
        assert len(email.sent) == 1                       # email STILL sent
        assert result.ok is True                          # all real edges → succeeded
        assert result.metrics["narrative_degraded"] is True
        _, body = email.sent[0]
        assert "Narrative unavailable" in body
        assert "LEAKYKEY" not in body                     # no leaked secret in the email

    def test_smtp_failure_fails_with_redacted_cause(self, base_config):
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail(raise_exc=BriefingEmailError(_SECRET_TOKEN))
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is False
        assert {e.code for e in result.errors} == {"smtp_send_failed"}
        cause = result.errors[0].cause or ""
        assert "SUPERSECRET" not in cause                 # §L10 redaction
        assert "***" in cause


# ---------------------------------------------------------------------------
# §N5 / §S5 — email-delivery idempotency
# ---------------------------------------------------------------------------
class TestDeliveryIdempotency:
    def test_already_delivered_skips_send_and_llm(self, base_config):
        """A prior delivery row → succeeded, NO send, NO LLM call (the check is
        before the LLM/render so a re-run is cheap)."""
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        llm = _FakeLlm()
        existing = BriefingDeliveryRow(
            briefing_day_utc=date(2026, 5, 26),
            model_version=_VERSION,
            run_id=uuid4(),
            sent_at=_NOW,
        )
        delivery = _FakeDelivery(existing=existing)
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
            llm=llm,
            delivery=delivery,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is True
        assert result.errors == ()
        assert email.sent == []                       # NOT re-sent
        assert llm.calls == 0                         # LLM not called on a re-run
        assert delivery.recorded == []                # nothing newly recorded
        assert result.metrics["already_delivered"] is True
        assert result.metrics["email_sent"] is False

    def test_first_send_records_delivery(self, base_config):
        """No prior row → send once → record one delivery keyed on the
        decisioning UTC day + active model_version, with the pinned sent_at."""
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        delivery = _FakeDelivery()
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            fmrows=[_fmrow(1)],
            email=email,
            delivery=delivery,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is True
        assert len(email.sent) == 1
        assert len(delivery.recorded) == 1
        row = delivery.recorded[0]
        assert row.briefing_day_utc == date(2026, 5, 26)
        assert row.model_version == _VERSION
        assert row.sent_at == _NOW

    def test_record_failure_is_best_effort(self, base_config):
        """A post-send record failure must NOT flip a genuinely-sent briefing to
        failed — the email went out, so the run stays succeeded (§S5)."""
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        delivery = _FakeDelivery(record_raise=StorageError(_SECRET_DSN))
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            fmrows=[_fmrow(1)],
            email=email,
            delivery=delivery,
        )
        result = agent.run(_ctx(base_config))   # must NOT raise
        assert result.ok is True
        assert len(email.sent) == 1             # sent exactly once

    def test_get_storage_error_fails_closed_no_send(self, base_config):
        """A DB error on the pre-send idempotency check fails closed:
        briefing_db_error, no send (never risk a double-send on an
        indeterminate DB)."""
        matches, players, tourns = _full_slate([1])
        email = _FakeEmail()
        delivery = _FakeDelivery(get_raise=StorageError(_SECRET_DSN))
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1)],
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
            delivery=delivery,
        )
        result = agent.run(_ctx(base_config))
        assert result.ok is False
        assert {e.code for e in result.errors} == {"briefing_db_error"}
        assert email.sent == []


# ---------------------------------------------------------------------------
# Active-version filter + slate window + isolation
# ---------------------------------------------------------------------------
class TestWiring:
    def test_filters_to_active_model_version(self, base_config):
        matches, players, tourns = _full_slate([1, 2])
        email = _FakeEmail()
        repo = _FakePredictionRepo(
            [_qualifying(1, model_version=_VERSION),
             _qualifying(2, model_version="OLD-MODEL")]
        )
        agent = _make_agent(
            base_config,
            active=_model_row(version=_VERSION),
            prediction_repo=repo,
            matches=matches,
            players=players,
            tourns=tourns,
            email=email,
        )
        result = agent.run(_ctx(base_config))
        # the read was filtered to the active version...
        assert repo.calls[0][0] == _VERSION
        # ...and only the active-version match surfaced.
        assert result.metrics["n_surfaced"] == 1

    def test_slate_window_is_midnight_utc_day(self, base_config):
        matches, players, tourns = _full_slate([1])
        repo = _FakePredictionRepo([_qualifying(1)])
        agent = _make_agent(
            base_config,
            active=_model_row(),
            prediction_repo=repo,
            matches=matches,
            players=players,
            tourns=tourns,
        )
        agent.run(_ctx(base_config))
        _, since, until = repo.calls[0]
        assert since == datetime(2026, 5, 26, 0, 0, tzinfo=UTC)
        assert until == datetime(2026, 5, 27, 0, 0, tzinfo=UTC)

    def test_per_match_context_error_dead_lettered(self, base_config):
        """§L2 — one match's repo read raises → dead-lettered + skipped; the
        other still surfaces; the run is partial and the cause is redacted."""
        matches, players, tourns = _full_slate([1, 2])
        email = _FakeEmail()
        dead = _FakeDeadLetter()
        match_repo = _FakeMatchRepo(matches, raise_for={1})
        agent = _make_agent(
            base_config,
            active=_model_row(),
            preds=[_qualifying(1), _qualifying(2)],
            players=players,
            tourns=tourns,
            email=email,
            dead=dead,
            match_repo=match_repo,
        )
        result = agent.run(_ctx(base_config))
        assert len(email.sent) == 1
        assert result.ok is False  # dead-lettered → partial
        assert result.metrics["n_surfaced"] == 1
        assert result.metrics["n_dead_lettered"] == 1
        assert len(dead.rows) == 1
        cause = dead.rows[0].error.get("cause", "")
        assert "topsecret" not in cause
        assert dead.rows[0].source == "briefing"


# ---------------------------------------------------------------------------
# Edge classification — per-side Shin→proportional fallback (Codex B1 F1)
# ---------------------------------------------------------------------------
class TestEdgeClassification:
    def test_shin_preferred_when_present(self):
        c = _classify(
            _pred(1, edge1=0.05, edge2=-0.04, prop1=0.03, prop2=-0.02), threshold=0.01
        )
        assert c.method == "shin"
        assert c.edge_p1 == 0.05 and c.edge_p2 == -0.04
        assert c.qualifying is True and c.no_market is False

    def test_proportional_fallback_when_shin_absent(self):
        c = _classify(_pred(1, prop1=0.05, prop2=-0.04), threshold=0.01)
        assert c.method == "proportional"
        assert c.edge_p1 == 0.05
        assert c.qualifying is True and c.no_market is False

    def test_below_threshold_not_qualifying(self):
        c = _classify(_pred(1, edge1=0.005, edge2=-0.005), threshold=0.01)
        assert c.qualifying is False and c.no_market is False

    def test_no_edges_is_no_market(self):
        c = _classify(_pred(1), threshold=0.01)
        assert c.no_market is True and c.qualifying is False
        assert c.edge_p1 is None and c.edge_p2 is None

    def test_mixed_side_fallback_does_not_drop_value_side(self):
        """Codex F1 regression: p1 Shin degenerate (None) but has a proportional
        value; p2 Shin present. Per-side fallback MUST surface p1's 0.05 (the
        value side), not drop it — the old per-row selection scored -0.04 and
        wrongly failed qualification."""
        row = _pred(1, edge1=None, edge2=-0.04, prop1=0.05, prop2=None)
        c = _classify(row, threshold=0.01)
        assert c.edge_p1 == 0.05      # value side preserved via per-side fallback
        assert c.edge_p2 == -0.04     # Shin used where present
        assert c.qualifying is True
        assert c.no_market is False


# ---------------------------------------------------------------------------
# Pipeline status mapping (§N4 fatal-code wiring)
# ---------------------------------------------------------------------------
class TestPipelineMapping:
    def test_briefing_fatal_codes_registered(self):
        for code in ("no_qualifying_predictions", "smtp_send_failed", "briefing_db_error"):
            assert code in _FATAL_CODES
        # partial code must NOT be fatal
        assert "briefing_partial" not in _FATAL_CODES

    @pytest.mark.parametrize(
        "result,expected",
        [
            (AgentResult(ok=True, metrics={}, errors=()), "succeeded"),
            (
                AgentResult(
                    ok=False,
                    metrics={},
                    errors=(AgentError(code="briefing_partial", message="x"),),
                ),
                "partial",
            ),
            (
                AgentResult(
                    ok=False,
                    metrics={},
                    errors=(AgentError(code="no_qualifying_predictions", message="x"),),
                ),
                "failed",
            ),
            (
                AgentResult(
                    ok=False,
                    metrics={},
                    errors=(AgentError(code="smtp_send_failed", message="x"),),
                ),
                "failed",
            ),
        ],
    )
    def test_map_status(self, result, expected):
        assert DailyPipeline._map_status(result) == expected
