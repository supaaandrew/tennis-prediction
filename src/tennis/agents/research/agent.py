"""ResearchAgent — the second pipeline agent (R6a).

Assembles the nine R3–R7 feature families behind one agent, builds each
match's point-in-time `FeatureContext`, runs every extractor, validates the
merged matrix (C10), and writes `feature_matrix` rows. It is the first agent
with a non-empty precondition (`data` must have succeeded for the run — §AGENTS).

Two scopes, selected by an explicit ctor `mode` (the `Agent` Protocol fixes
`run(ctx)`, so scope cannot be a `run()` argument — it is a wiring decision, like
DataAgent's injected config):

  - ``"training"``  — source `final` matches (C4 `for_training`), run `EloWalk`
    ONCE over them (§M9 single-shot: it populates the append-only `elo_snapshots`
    ladder AND returns the per-match Elo fragments carrying the correct
    count-before-this-match `reliability_low`); those fragments ARE the `elo`
    family. The history index is these finals.
  - ``"prediction"`` — produce rows for the upcoming slate (C4 `for_prediction`,
    `scheduled`/`live`). The history the form/h2h/surface/serve families read is
    the SAME historical `final` record (a scheduled match's priors are past
    finals), so the index is still built from `for_training`; only the LOOP set
    is the slate. Elo career counts are reconstructed from the already-built
    ladder via `EloSnapshotRepository.career_match_counts()` (the §M9 counter is
    in-memory only during the walk) and an `EloExtractor` reads the ladder.

PIT discipline: `as_of_ts` comes only from `pit_cut`; the matrix is validated
before any write; `feature_matrix` stores CLEAN values (H1 — noise is a Modeling
concern). Per-match fault isolation mirrors DataAgent (§L2/§L10): one bad match
is dead-lettered and skipped (→ `partial`), never aborting the slate; a rejected
matrix is a hard C10 stop (→ `failed`), writing nothing.

Extensibility (§M4): the eight ordinary per-match families are built from an
ordered `extractor_factories` registry the agent iterates; R7 (fatigue/market)
appended its factories without touching run()/merge/validate. Elo is the one
mode-special family.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tennis.adapters.matchstat.sidecars import MatchstatH2hStatsClient

from tennis.core.config import AppConfig
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import FeatureContractError, FeatureMatrixValidationError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy, Precondition
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    EloSnapshotRepository,
    FeatureMatrixRepository,
    FeatureSpecRepository,
    MatchRepository,
    MatchStatRepository,
    OddsSnapshotRepository,
    PlayerRankingRepository,
    TournamentRepository,
    VenueRepository,
    WeatherObservationRepository,
)
from tennis.storage.postgres.rows import DeadLetterRow, FeatureMatrixRow, MatchRow
from tennis.agents.research import specs
from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.conditions import ConditionsExtractor
from tennis.agents.research.features.elo import EloExtractor, EloWalk
from tennis.agents.research.features.fatigue import FatigueExtractor
from tennis.agents.research.features.form import FormExtractor
from tennis.agents.research.features.h2h import H2HExtractor
from tennis.agents.research.features.market import MarketExtractor
from tennis.agents.research.features.rankings import RankingsExtractor
from tennis.agents.research.features.serve_return import ServeReturnExtractor
from tennis.agents.research.features.surface import SurfaceExtractor
from tennis.agents.research.point_in_time import pit_cut
from tennis.agents.research.validator import FeatureMatrixValidator

_logger = get_logger("tennis.agents.research.agent")

ResearchMode = Literal["training", "prediction"]

# The `elo` family is mode-special (walk in training, extractor in prediction),
# so it is handled directly by the agent, never via the factory registry.
ELO_FAMILY = "elo"

# Heartbeat cadence inside the per-match loop (§L7). Beats fire between batches so
# a long historical pass cannot look orphaned; the exact period is not load-bearing.
_HEARTBEAT_EVERY = 50


@dataclass(frozen=True, slots=True)
class _ExtractorDeps:
    """The per-run dependencies an ordinary extractor factory may pick from.

    Built once inside `run()` after the `MatchHistoryIndex` exists (the history
    families can't be constructed at wiring time — the index is per-run)."""

    history: MatchHistoryIndex
    config: AppConfig
    ranking_repo: PlayerRankingRepository
    tournament_repo: TournamentRepository
    stat_repo: MatchStatRepository
    weather_repo: WeatherObservationRepository
    venue_repo: VenueRepository
    odds_repo: OddsSnapshotRepository
    # §T14 — optional shared matchstat h2h-stats client. The sidecar's per-pair
    # cache must live across the full per-match loop (built once per run by
    # composition and shared with the H2HExtractor factory).
    h2h_stats_client: "MatchstatH2hStatsClient | None" = None


ExtractorFactory = Callable[[_ExtractorDeps], FeatureExtractor]

# Ordered registry of the eight ordinary per-match families (everything except the
# mode-special `elo`). R7 appended fatigue + market here — or the DI layer injects an
# extended list via the ctor — and `run()` is unchanged (§M4). The active family set used for
# seeding/validation is derived from whatever list the instance holds, so the
# catalog (specs._REGISTRY) and the extractors stay in lockstep (§M7).
_ORDINARY_FAMILY_FACTORIES: tuple[tuple[str, ExtractorFactory], ...] = (
    ("rankings", lambda d: RankingsExtractor(ranking_repo=d.ranking_repo, config=d.config)),
    ("form", lambda d: FormExtractor(history=d.history, config=d.config)),
    ("h2h", lambda d: H2HExtractor(
        history=d.history, tournament_repo=d.tournament_repo, config=d.config,
        h2h_stats_client=d.h2h_stats_client)),
    ("surface", lambda d: SurfaceExtractor(
        history=d.history, tournament_repo=d.tournament_repo, config=d.config)),
    ("serve_return", lambda d: ServeReturnExtractor(
        history=d.history, stat_repo=d.stat_repo, config=d.config)),
    ("conditions", lambda d: ConditionsExtractor(
        weather_repo=d.weather_repo, venue_repo=d.venue_repo, config=d.config)),
    # R7 — additive (§M4). Fatigue also needs tournament_repo (to resolve a PRIOR
    # match's venue for travel-km); market needs the new odds_repo dep.
    ("fatigue", lambda d: FatigueExtractor(
        history=d.history, tournament_repo=d.tournament_repo,
        venue_repo=d.venue_repo, config=d.config)),
    ("market", lambda d: MarketExtractor(odds_repo=d.odds_repo, config=d.config)),
)


class ResearchAgent:
    """Implements the `Agent` Protocol: `name`, `lineage`, `run(ctx)`."""

    name = "research"

    def __init__(
        self,
        *,
        mode: ResearchMode,
        config: AppConfig,
        match_repo: MatchRepository,
        elo_repo: EloSnapshotRepository,
        tournament_repo: TournamentRepository,
        ranking_repo: PlayerRankingRepository,
        stat_repo: MatchStatRepository,
        weather_repo: WeatherObservationRepository,
        venue_repo: VenueRepository,
        odds_repo: OddsSnapshotRepository,
        feature_spec_repo: FeatureSpecRepository,
        feature_matrix_repo: FeatureMatrixRepository,
        dead_letter: DeadLetterRepository,
        extractor_factories: Sequence[tuple[str, ExtractorFactory]] = _ORDINARY_FAMILY_FACTORIES,
        h2h_stats_client: "MatchstatH2hStatsClient | None" = None,
    ) -> None:
        if mode not in ("training", "prediction"):
            raise ValueError(f"ResearchAgent: unknown mode {mode!r}")
        self._mode: ResearchMode = mode
        self._config = config
        self._match_repo = match_repo
        self._elo_repo = elo_repo
        self._tournament_repo = tournament_repo
        self._ranking_repo = ranking_repo
        self._stat_repo = stat_repo
        self._weather_repo = weather_repo
        self._venue_repo = venue_repo
        self._odds_repo = odds_repo
        self._feature_spec_repo = feature_spec_repo
        self._feature_matrix_repo = feature_matrix_repo
        self._dead_letter = dead_letter
        self._factories = tuple(extractor_factories)
        # §T14 — optional matchstat clutch H2H. None for training-only chains
        # (no key) — the H2HExtractor then emits all 6 clutch keys as NULL.
        self._h2h_stats_client = h2h_stats_client
        # Active families = elo + the ordinary factories, in catalog order. Drives
        # both seeding and the validator's expected_specs so they never diverge.
        self._families: tuple[str, ...] = (ELO_FAMILY,) + tuple(
            name for name, _ in self._factories
        )
        self._feature_set = config.features.feature_set

        hb = config.orchestrator.heartbeat
        # First agent with a real precondition: `data` must have succeeded for
        # this run before Research may start (§AGENTS / the §R6 lineage chain).
        self.lineage = AgentLineage(
            preconditions=(
                Precondition(previous_agent="data", required_status="succeeded"),
            ),
            heartbeat=HeartbeatPolicy(
                interval_s=hb.interval_s, orphan_after_s=hb.orphan_after_s
            ),
        )
        # §M12 windows-lockstep is a deploy-time config invariant — validate at
        # CONSTRUCTION so a misconfigured agent is never wired into a run (no
        # dangling 'running' row) and run() always honors the Agent contract
        # (returns AgentResult, never raises). Fail loud here.
        self._assert_windows_in_lockstep()

    # -- entrypoint ---------------------------------------------------------
    def run(self, ctx: AgentContext) -> AgentResult:
        # --- Startup, BEFORE the match loop (control flow: seed →
        # build expected_specs → loop → validate → upsert; the §M12 windows
        # guard already ran at construction). ---
        specs.seed_feature_specs(self._feature_spec_repo, families=self._families)
        # build_expected_specs reads the just-seeded catalog and hard-fails on
        # drift — so it MUST run after the seed, never after the write.
        expected = specs.build_expected_specs(
            self._feature_spec_repo,
            feature_set=self._feature_set,
            families=self._families,
        )

        # The history substrate is ALWAYS the `final` record (a prediction
        # match's priors are past finals), so build the index from for_training
        # in both modes; only the loop set differs (§AGENTS: inputs are BOTH
        # for_training and for_prediction).
        rng = self._config.ingestion.history_backfill_season_range
        history_matches = list(
            self._match_repo.for_training(season_start=rng.start, season_end=rng.end)
        )
        index = MatchHistoryIndex.build(history_matches)
        extractors = self._build_extractors(index)

        # Elo is mode-special: training emits the walk's per-match fragments;
        # prediction reconstructs the §M9 counts and reads the built ladder.
        ctx.heartbeat()
        elo_fragments, elo_extractor, elo_metrics = self._prepare_elo(history_matches)

        if self._mode == "training":
            loop_matches = history_matches
        else:
            loop_matches = list(
                self._match_repo.for_prediction(
                    as_of=ctx.as_of,
                    lookforward_days=self._config.ingestion.daily_lookforward_days,
                )
            )

        # --- Per-match extraction with fault isolation. ---
        offset = self._config.decision_timing.live_decision_offset_hours
        rows: list[dict[str, Any]] = []
        match_starts: dict[int, Any] = {}
        match_dates: dict[int, Any] = {}
        dead = 0

        for i, match in enumerate(loop_matches):
            if i % _HEARTBEAT_EVERY == 0:
                ctx.heartbeat()
            try:
                row = self._extract_one(
                    match, offset=offset,
                    extractors=extractors,
                    elo_fragments=elo_fragments,
                    elo_extractor=elo_extractor,
                    ctx=ctx,
                )
            except Exception as exc:  # per-match isolation (§L2) — never abort
                self._dead_letter_match(
                    ctx, match, reason="extraction_error", cause=_safe_cause(exc)
                )
                dead += 1
                continue
            if row is None:  # tournament unresolved / elo fragment missing — skipped
                dead += 1
                continue
            rows.append(row)
            match_starts[match.match_id] = match.start_ts
            match_dates[match.match_id] = match.match_date

        metrics: dict[str, Any] = {
            "mode": self._mode,
            "elo": elo_metrics,
            "scope": {
                "matches_total": len(loop_matches),
                "matches_dead_lettered": dead,
                "matches_written": 0,
            },
        }

        # --- Validate the WHOLE matrix BEFORE any write (C10). ---
        validator = FeatureMatrixValidator(
            expected_specs=expected,
            match_starts=match_starts,
            match_dates=match_dates,
        )
        try:
            validator.validate(rows=rows)
        except FeatureMatrixValidationError as exc:
            # A rejected matrix is a hard stop: write nothing (the Research→
            # Modeling gate). "feature_matrix_invalid" is a fatal code → 'failed'.
            _logger.error("research_matrix_rejected", violations=len(exc.violations))
            return AgentResult(
                ok=False,
                metrics=metrics,
                errors=(
                    AgentError(
                        code="feature_matrix_invalid",
                        message="feature matrix validation rejected the slate",
                        cause=redact_text(str(exc)),
                    ),
                ),
            )

        # --- Persist (catalog already seeded at startup). CLEAN values (H1). ---
        # A `fm_no_lookahead` trigger violation MUST propagate, never be swallowed.
        for row in rows:
            self._feature_matrix_repo.upsert(
                FeatureMatrixRow(
                    match_id=row["match_id"],
                    feature_set=row["feature_set"],
                    as_of_ts=row["as_of_ts"],
                    payload=row["payload"],
                )
            )
        metrics["scope"]["matches_written"] = len(rows)

        ok = dead == 0
        _logger.info(
            "research_run_complete",
            mode=self._mode, written=len(rows), dead_lettered=dead, ok=ok,
        )
        return AgentResult(ok=ok, metrics=metrics, errors=())

    # -- internals ----------------------------------------------------------
    def _build_extractors(self, index: MatchHistoryIndex) -> list[FeatureExtractor]:
        deps = _ExtractorDeps(
            history=index,
            config=self._config,
            ranking_repo=self._ranking_repo,
            tournament_repo=self._tournament_repo,
            stat_repo=self._stat_repo,
            weather_repo=self._weather_repo,
            venue_repo=self._venue_repo,
            odds_repo=self._odds_repo,
            h2h_stats_client=self._h2h_stats_client,
        )
        return [factory(deps) for _, factory in self._factories]

    def _prepare_elo(
        self, history_matches: list[MatchRow]
    ) -> tuple[Mapping[int, Mapping[str, Any]] | None, EloExtractor | None, dict[str, Any]]:
        """Mode-special Elo wiring. Training: run the single-shot walk (§M9) for
        per-match fragments + the ladder. Prediction: reconstruct the §M9 counts
        from the built ladder and read it via `EloExtractor`."""
        if self._mode == "training":
            walk = EloWalk(
                elo_repo=self._elo_repo,
                tournament_repo=self._tournament_repo,
                config=self._config,
            )
            fragments = walk.run(history_matches)
            metrics = {
                "fragments": len(fragments),
                "skipped_invalid_winner": walk.skipped_invalid_winner,
            }
            return fragments, None, metrics
        # prediction
        career_counts = self._elo_repo.career_match_counts()
        extractor = EloExtractor(
            elo_repo=self._elo_repo,
            config=self._config,
            career_counts=career_counts,
        )
        return None, extractor, {"career_counts_players": len(career_counts)}

    def _extract_one(
        self,
        match: MatchRow,
        *,
        offset: int,
        extractors: list[FeatureExtractor],
        elo_fragments: Mapping[int, Mapping[str, Any]] | None,
        elo_extractor: EloExtractor | None,
        ctx: AgentContext,
    ) -> dict[str, Any] | None:
        """Build one match's merged feature row, or None to skip+dead-letter it.

        Returns a plain dict (the shape `FeatureMatrixValidator.validate` reads via
        subscript); a `FeatureMatrixRow` is constructed only at write time."""
        tournament = self._tournament_repo.get(match.tournament_id)
        if tournament is None:
            # Without a surface the critical base-Elo keys can't be emitted, so
            # the row would fail validator R3 anyway — skip + dead-letter (decision 2).
            self._dead_letter_match(ctx, match, reason="tournament_unresolved")
            return None

        as_of = pit_cut(match, live_offset_hours=offset)
        fctx = FeatureContext(
            match=match,
            as_of_ts=as_of,
            feature_set=self._feature_set,
            surface=tournament.surface,
            indoor=tournament.indoor,
            venue_id=tournament.venue_id,
            tier=tournament.tier,
        )

        payload: dict[str, Any] = {}
        for ext in extractors:
            payload.update(ext.extract(fctx))

        if self._mode == "training":
            assert elo_fragments is not None
            fragment = elo_fragments.get(match.match_id)
            if fragment is None:
                # The walk skipped this match (surface unresolved) though the
                # tournament resolved here — inconsistent; skip rather than emit
                # a row missing the critical Elo keys.
                self._dead_letter_match(ctx, match, reason="elo_fragment_missing")
                return None
            payload.update(fragment)
        else:
            assert elo_extractor is not None
            payload.update(elo_extractor.extract(fctx))

        return {
            "match_id": match.match_id,
            "feature_set": self._feature_set,
            "as_of_ts": as_of,
            "payload": payload,
        }

    def _assert_windows_in_lockstep(self) -> None:
        """§M12 construction-time guard: the runtime Form windows (`config.
        features.windows_days`, what `FormExtractor` emits) MUST equal the PINNED
        windows the Form catalog was seeded from (`specs._FORM_WINDOWS`). A
        config-only change deployed without CI would otherwise diverge at runtime
        — fewer windows → loud R1 failure; EXTRA windows → silently emitted,
        unvalidated. Run from `__init__` so a misconfigured agent fails at wiring
        (no run row stranded) rather than mid-run."""
        runtime = tuple(self._config.features.windows_days)
        seeded = tuple(specs._FORM_WINDOWS)
        if runtime != seeded:
            raise FeatureContractError(
                "config.features.windows_days "
                f"{runtime} does not match the seeded Form catalog windows "
                f"{seeded} (§M12 lockstep)"
            )

    def _dead_letter_match(
        self,
        ctx: AgentContext,
        match: MatchRow,
        *,
        reason: str,
        cause: str | None = None,
    ) -> None:
        """Record a skipped match and continue (§L2). `append` never raises."""
        _logger.warning(
            "research_match_skipped",
            match_id=match.match_id, tournament_id=match.tournament_id, reason=reason,
        )
        error: dict[str, Any] = {"reason": reason}
        if cause is not None:
            error["cause"] = cause
        self._dead_letter.append(
            DeadLetterRow(
                payload={"match_id": match.match_id, "tournament_id": match.tournament_id},
                error=error,
                run_id=ctx.run_id,
                source="research",
                scope="feature_matrix",
            )
        )


def _safe_cause(exc: BaseException) -> str:
    """Exception type + a credential-redacted message — never raw `repr(exc)`,
    which can carry a token-bearing message into `dead_letter.error` or the logs
    (mirrors DataAgent §L10)."""
    return f"{type(exc).__name__}: {redact_text(str(exc))}"
