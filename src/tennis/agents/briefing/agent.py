"""BriefingAgent — the fourth and final pipeline agent (B1).

Reads the day's `predictions` (written by `ModelingAgent(mode="prediction")`),
asks Claude for a narrative, renders an edge/Kelly email, and sends it. It
**consumes** M1b's output — it never recomputes edges or Kelly (§M24).

Flow (`run` never raises for the documented failure modes — it returns a
`failed`/`partial`/`succeeded` `AgentResult`):

  1. Resolve the active model (`None` → `failed`, zero email).
  2. Read the decisioning-day slate (§N3 midnight-UTC window) filtered to the
     ACTIVE `model_version`.
  3. Classify each row by its primary (Shin → proportional fallback) edge:
     qualifying (`>= modeling.edge.min_edge_to_log`) ∪ no-market (C9 NULL edges).
     No qualifying rows → `failed`, zero email (§N4 / AGENTS.md).
  4. Enrich each surfaced row with player/tournament names (per-match fault
     isolation, §L2; a missing reference row degrades to a fallback label).
  5. Build the prompt → ONE LLM call → narrative. An LLM failure is isolated +
     redacted and NON-fatal (§N4): the email still sends, narrative-less.
  6. Render (C13 disclaimer on every Kelly line; closing fields never rendered,
     §M19/§15.4) and send. SMTP failure → `failed` (§L10 redacted cause).

Status (§N4): `succeeded` = sent, all surfaced rows had real edges + nothing
dead-lettered; `partial` = sent but ≥1 surfaced row was no-market (C9) or some
row was dead-lettered; `failed` = no active model / no qualifying rows / SMTP
failure / DB unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from uuid import UUID

from tennis.core.config import AppConfig
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import StorageError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy, Precondition
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import (
    BriefingDeliveryRepository,
    DeadLetterRepository,
    FeatureMatrixRepository,
    MatchRepository,
    ModelRegistryRepository,
    PlayerRepository,
    PredictionRepository,
    TournamentRepository,
)
from tennis.storage.postgres.rows import (
    BriefingDeliveryRow,
    DeadLetterRow,
    PredictionRow,
)

from tennis.agents.briefing.email_client import EmailClient
from tennis.agents.briefing.llm_client import LlmClient
from tennis.agents.briefing.render import SurfacedMatch, render_email

_logger = get_logger("tennis.agents.briefing.agent")


class BriefingAgent:
    """Implements the `Agent` Protocol: `name`, `lineage`, `run(ctx)`."""

    name = "briefing"

    def __init__(
        self,
        *,
        config: AppConfig,
        model_registry_repo: ModelRegistryRepository,
        prediction_repo: PredictionRepository,
        match_repo: MatchRepository,
        player_repo: PlayerRepository,
        tournament_repo: TournamentRepository,
        feature_matrix_repo: FeatureMatrixRepository,
        dead_letter_repo: DeadLetterRepository,
        briefing_delivery_repo: BriefingDeliveryRepository,
        llm_client: LlmClient,
        email_client: EmailClient,
    ) -> None:
        self._config = config
        self._model_registry_repo = model_registry_repo
        self._prediction_repo = prediction_repo
        self._match_repo = match_repo
        self._player_repo = player_repo
        self._tournament_repo = tournament_repo
        self._feature_matrix_repo = feature_matrix_repo
        self._dead_letter_repo = dead_letter_repo
        self._briefing_delivery_repo = briefing_delivery_repo
        self._llm_client = llm_client
        self._email_client = email_client
        self._feature_set_name = config.features.feature_set

        hb = config.orchestrator.heartbeat
        self.lineage = AgentLineage(
            preconditions=(
                Precondition(previous_agent="modeling", required_status="succeeded"),
            ),
            heartbeat=HeartbeatPolicy(
                interval_s=hb.interval_s, orphan_after_s=hb.orphan_after_s
            ),
        )

    # -- entrypoint ---------------------------------------------------------
    def run(self, ctx: AgentContext) -> AgentResult:
        try:
            return self._run(ctx)
        except StorageError as exc:
            _logger.error("briefing_db_error", cause=redact_text(str(exc)))
            return AgentResult(
                ok=False,
                metrics={"stage": "io", "email_sent": False},
                errors=(
                    AgentError(
                        code="briefing_db_error",
                        message="database unavailable during briefing",
                        cause=redact_text(str(exc)),
                    ),
                ),
            )

    def _run(self, ctx: AgentContext) -> AgentResult:
        ctx.heartbeat()
        active = self._model_registry_repo.active()
        if active is None:
            _logger.error("briefing_no_active_model")
            return AgentResult(
                ok=False,
                metrics={"n_surfaced": 0, "email_sent": False},
                errors=(
                    AgentError(
                        code="no_active_model",
                        message="no active model in the registry; nothing to brief",
                    ),
                ),
            )

        since, until = _slate_window(ctx.as_of)
        rows = list(
            self._prediction_repo.list_for_window(
                model_version=active.version, since=since, until=until
            )
        )
        threshold = self._config.modeling.edge.min_edge_to_log
        classified = [_classify(r, threshold=threshold) for r in rows]
        qualifying = [c for c in classified if c.qualifying]
        surfaced_classified = [c for c in classified if c.qualifying or c.no_market]

        if not qualifying:
            _logger.info("briefing_no_qualifying", n_slate=len(rows))
            return AgentResult(
                ok=False,
                metrics={
                    "model_version": active.version,
                    "n_slate": len(rows),
                    "n_surfaced": 0,
                    "email_sent": False,
                },
                errors=(
                    AgentError(
                        code="no_qualifying_predictions",
                        message=(
                            "no predictions at/above min_edge_to_log; no briefing sent"
                        ),
                    ),
                ),
            )

        # §N5/§S5: durable email-delivery idempotency. A prior run that already
        # delivered this day's briefing for this model leaves a row; a manual
        # re-run or a crash-after-send must NOT re-send. Checked here — before
        # the payload reads, LLM call, render, and send — so a re-run is cheap.
        # A StorageError propagates to `run()` → `briefing_db_error` (fatal, no
        # send): fail-closed, never double-send on an indeterminate DB.
        briefing_day = ctx.as_of.date()
        if (
            self._briefing_delivery_repo.get(
                briefing_day_utc=briefing_day, model_version=active.version
            )
            is not None
        ):
            _logger.info(
                "briefing_already_delivered",
                briefing_day_utc=briefing_day.isoformat(),
                model_version=active.version,
            )
            return AgentResult(
                ok=True,
                metrics={
                    "model_version": active.version,
                    "n_slate": len(rows),
                    "n_qualifying": len(qualifying),
                    "n_surfaced": 0,
                    "already_delivered": True,
                    "email_sent": False,
                },
                errors=(),
            )

        ctx.heartbeat()
        payloads = self._payloads_for([c.row.match_id for c in surfaced_classified])
        surfaced: list[SurfacedMatch] = []
        dead = 0
        for c in surfaced_classified:
            try:
                surfaced.append(self._build_surfaced(c))
            except Exception as exc:  # per-match isolation (§L2)
                self._dead_letter(
                    ctx,
                    match_id=c.row.match_id,
                    reason="briefing_context_error",
                    cause=_safe_cause(exc),
                )
                dead += 1

        if not surfaced:
            # Every qualifying row failed context resolution — nothing to render.
            return AgentResult(
                ok=False,
                metrics={
                    "model_version": active.version,
                    "n_slate": len(rows),
                    "n_qualifying": len(qualifying),
                    "n_surfaced": 0,
                    "n_dead_lettered": dead,
                    "email_sent": False,
                },
                errors=(
                    AgentError(
                        code="no_qualifying_predictions",
                        message="all qualifying rows failed context resolution",
                    ),
                ),
            )

        ctx.heartbeat()
        narrative, narrative_degraded = self._narrate(
            surfaced, payloads, as_of=ctx.as_of, model_version=active.version
        )

        subject, body = render_email(
            matches=surfaced,
            narrative=narrative,
            kelly_disclaimer=self._config.briefing.kelly_disclaimer,
            subject_template=self._config.briefing.email.subject_template,
            as_of=ctx.as_of,
            model_version=active.version,
        )

        try:
            self._email_client.send(subject=subject, body=body)
        except Exception as exc:  # BriefingEmailError + any escape → failed (§N4)
            cause = _safe_cause(exc)
            _logger.error("briefing_smtp_failed", cause=cause)
            return AgentResult(
                ok=False,
                metrics={
                    "model_version": active.version,
                    "n_slate": len(rows),
                    "n_qualifying": len(qualifying),
                    "n_surfaced": len(surfaced),
                    "n_dead_lettered": dead,
                    "email_sent": False,
                },
                errors=(
                    AgentError(
                        code="smtp_send_failed",
                        message="briefing email could not be delivered",
                        cause=cause,
                    ),
                ),
            )

        # §N5/§S5: record the delivery AFTER a confirmed send. Best-effort — a
        # failure here must NOT flip a genuinely-sent briefing to `failed`; the
        # narrow crash-between-send-and-record window is the accepted residual.
        self._record_delivery(
            briefing_day=briefing_day,
            model_version=active.version,
            run_id=ctx.run_id,
            sent_at=ctx.as_of,
        )

        n_no_market = sum(1 for m in surfaced if m.no_market)
        metrics: dict[str, Any] = {
            "model_version": active.version,
            "n_slate": len(rows),
            "n_qualifying": len(qualifying),
            "n_no_market": n_no_market,
            "n_surfaced": len(surfaced),
            "n_dead_lettered": dead,
            "narrative_degraded": narrative_degraded,
            "email_sent": True,
        }
        _logger.info("briefing_complete", **metrics)

        # `partial` (ok=False, NON-fatal error) when some surfaced rows had no
        # market (C9) or some row was dead-lettered; otherwise `succeeded`.
        if n_no_market > 0 or dead > 0:
            return AgentResult(
                ok=False,
                metrics=metrics,
                errors=(
                    AgentError(
                        code="briefing_partial",
                        message=(
                            f"sent with {n_no_market} no-market row(s), "
                            f"{dead} dead-lettered"
                        ),
                    ),
                ),
            )
        return AgentResult(ok=True, metrics=metrics, errors=())

    # -- helpers ------------------------------------------------------------
    def _record_delivery(
        self,
        *,
        briefing_day: date,
        model_version: str,
        run_id: UUID,
        sent_at: datetime,
    ) -> None:
        """Persist the delivery marker AFTER a confirmed send (§N5/§S5).

        Best-effort: every exception (including a `StorageError`) is logged and
        swallowed — a record failure must not flip a genuinely-sent briefing to
        `failed`. The narrow window between SMTP-OK and this write is the accepted
        irreducible §S5 residual; a re-run in that window may re-send. The failure
        is logged at ERROR as a **pageable** `briefing_delivery_record_failed`
        signal so an operator can repair idempotency state before the next run.
        """
        try:
            self._briefing_delivery_repo.record(
                BriefingDeliveryRow(
                    briefing_day_utc=briefing_day,
                    model_version=model_version,
                    run_id=run_id,
                    sent_at=sent_at,
                )
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, never propagate
            # ERROR (not WARNING): the email WENT OUT but the idempotency marker
            # did not persist, so a rerun may re-send. Page on this (§S5).
            _logger.error(
                "briefing_delivery_record_failed",
                briefing_day_utc=briefing_day.isoformat(),
                model_version=model_version,
                cause=_safe_cause(exc),
            )

    def _payloads_for(self, match_ids: Sequence[int]) -> Mapping[int, Mapping[str, Any]]:
        """Bulk feature-payload read for narrative grounding (non-critical).

        A `StorageError` degrades the narrative (empty signals), never the run —
        the actionable edge/Kelly data already lives on the prediction rows."""
        try:
            rows = self._feature_matrix_repo.list_for_matches(
                match_ids=list(match_ids), feature_set=self._feature_set_name
            )
        except StorageError as exc:
            _logger.warning("briefing_feature_read_failed", cause=redact_text(str(exc)))
            return {}
        return {r.match_id: r.payload for r in rows}

    def _build_surfaced(self, c: _Classified) -> SurfacedMatch:
        """Resolve display fields for one surfaced row. Missing reference rows
        degrade to fallback labels (never raise); a repo read EXCEPTION
        propagates to the caller's per-match isolation."""
        row = c.row
        match = self._match_repo.get(row.match_id)
        if match is None:
            p1_name, p2_name = f"match {row.match_id} (p1)", "(p2)"
            tournament, surface, rnd = "unknown", "unknown", "unknown"
        else:
            p1_name = self._player_name(match.p1_id)
            p2_name = self._player_name(match.p2_id)
            tour = self._tournament_repo.get(match.tournament_id)
            tournament = tour.name if tour is not None else "unknown"
            surface = tour.surface if tour is not None else "unknown"
            rnd = match.round
        return SurfacedMatch(
            match_id=row.match_id,
            p1_name=p1_name,
            p2_name=p2_name,
            tournament=tournament,
            surface=surface,
            round=rnd,
            p1_prob_cal=row.p1_prob_cal,
            edge_p1=c.edge_p1,
            edge_p2=c.edge_p2,
            edge_method=c.method,
            kelly_fraction_p1=row.kelly_fraction_p1,
            kelly_fraction_p2=row.kelly_fraction_p2,
            no_market=c.no_market,
        )

    def _player_name(self, player_id: int) -> str:
        player = self._player_repo.get(player_id)
        return player.full_name if player is not None else f"player {player_id}"

    def _narrate(
        self,
        surfaced: Sequence[SurfacedMatch],
        payloads: Mapping[int, Mapping[str, Any]],
        *,
        as_of: datetime,
        model_version: str,
    ) -> tuple[str | None, bool]:
        """One LLM call. Any failure is isolated + redacted and NON-fatal (§N4):
        returns `(None, True)` so the email still renders without the narrative."""
        prompt = _build_prompt(
            surfaced, payloads, as_of=as_of, model_version=model_version
        )
        try:
            return self._llm_client.generate(prompt=prompt), False
        except Exception as exc:  # BriefingLlmError + any escape → degrade
            _logger.warning("briefing_llm_failed", cause=_safe_cause(exc))
            return None, True

    def _dead_letter(
        self,
        ctx: AgentContext,
        *,
        match_id: int,
        reason: str,
        cause: str | None = None,
    ) -> None:
        """Record a skipped match and continue (§L2). `append` never raises."""
        _logger.warning("briefing_dead_letter", match_id=match_id, reason=reason)
        error: dict[str, Any] = {"reason": reason}
        if cause is not None:
            error["cause"] = cause
        self._dead_letter_repo.append(
            DeadLetterRow(
                payload={"match_id": match_id},
                error=error,
                run_id=ctx.run_id,
                source="briefing",
                scope="briefing",
            )
        )


# ---------------------------------------------------------------------------
# Pure helpers (module-level — no agent state)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Classified:
    row: PredictionRow
    edge_p1: float | None
    edge_p2: float | None
    method: str  # "shin" | "proportional" | "none"
    no_market: bool
    qualifying: bool


def _classify(row: PredictionRow, *, threshold: float) -> _Classified:
    """Resolve each side's edge with a PER-SIDE Shin→proportional fallback
    (§M24), then decide qualifying (≥ threshold on the value side) vs no-market
    (C9).

    Per-side (not per-row) resolution: a row whose Shin devig degenerated on one
    side still surfaces the other side's value instead of dropping it. Today
    `compute_edges` writes the two Shin edges atomically (both-or-neither) and
    the two proportional edges atomically, so the mixed case cannot arise from
    the real writer — but the briefing reads `PredictionRow` decoupled from that
    writer and must not silently mis-handle a partial/future write (Codex B1 F1).
    """
    e1 = row.edge_p1_shin if row.edge_p1_shin is not None else row.edge_p1_proportional
    e2 = row.edge_p2_shin if row.edge_p2_shin is not None else row.edge_p2_proportional
    if row.edge_p1_shin is not None or row.edge_p2_shin is not None:
        method = "shin"
    elif e1 is not None or e2 is not None:
        method = "proportional"
    else:
        method = "none"
    no_market = e1 is None and e2 is None
    available = [e for e in (e1, e2) if e is not None]
    best = max(available) if available else None
    qualifying = best is not None and best >= threshold
    return _Classified(
        row=row,
        edge_p1=e1,
        edge_p2=e2,
        method=method,
        no_market=no_market,
        qualifying=qualifying,
    )


def _slate_window(as_of: datetime) -> tuple[datetime, datetime]:
    """§N3 — midnight-UTC of the decisioning day to the next midnight UTC."""
    if as_of.tzinfo is None:
        raise ValueError("_slate_window: as_of must be tz-aware")
    since = datetime.combine(as_of.date(), time.min, tzinfo=UTC)
    until = since + timedelta(days=1)
    return since, until


def _build_prompt(
    surfaced: Sequence[SurfacedMatch],
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    as_of: datetime,
    model_version: str,
) -> str:
    lines = [
        "You are writing a concise daily tennis edge briefing for a research "
        "audience. Summarize the value opportunities below in 2-4 short paragraphs.",
        "Be factual and measured. This is for research only — not betting advice. "
        "Do not invent statistics beyond what is given.",
        "",
        f"Date: {as_of.date().isoformat()}  Model: {model_version}",
        "",
        "Matches:",
    ]
    for m in surfaced:
        signal = _signal_snippet(payloads.get(m.match_id, {}))
        if m.no_market:
            lines.append(
                f"- {m.p1_name} vs {m.p2_name} ({m.tournament}, {m.surface}): "
                f"model {m.p1_name} {m.p1_prob_cal:.0%}; no market{signal}"
            )
        else:
            lines.append(
                f"- {m.p1_name} vs {m.p2_name} ({m.tournament}, {m.surface}): "
                f"model {m.p1_name} {m.p1_prob_cal:.0%}, "
                f"edge {m.p1_name} {_edge_pct(m.edge_p1)} / "
                f"{m.p2_name} {_edge_pct(m.edge_p2)}{signal}"
            )
    return "\n".join(lines)


def _signal_snippet(payload: Mapping[str, Any]) -> str:
    """A tiny, optional grounding signal — only the blended Elo gap if present."""
    elo = payload.get("elo_diff_blended")
    if isinstance(elo, (int, float)):
        return f", elo_diff_blended={float(elo):.2f}"
    return ""


def _edge_pct(edge: float | None) -> str:
    return "n/a" if edge is None else f"{edge * 100:+.1f}%"


def _safe_cause(exc: BaseException) -> str:
    """Exception type + a credential-redacted message — never raw `repr` (§L10)."""
    return f"{type(exc).__name__}: {redact_text(str(exc))}"
