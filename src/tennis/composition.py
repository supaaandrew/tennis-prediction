"""Composition root (§S7) — the single place the production object graph is
wired from `AppConfig` + environment secrets.

`build_training_chain` and `build_daily_chain` each return a fully-wired
`DailyPipeline` (an agent chain run under one `run_id`, §S2/§S3). The CLI calls
these; nothing else constructs the real graph. Secrets are read from the
environment here and validated up front (`validate_environment`); adapter network
egress and DB connections happen only when an agent RUNS, never at construction
(so these builders are unit-testable with dummy env + an injected session
factory, no network).

Layering: agents depend on repository Protocols + adapter factories; this module
is the one place the concrete `impl.py` repositories, the `HttpxClient`
transport, the source adapters, and the LLM/SMTP clients are named.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tennis.adapters.atp_scraper.adapter import AtpScraperAdapter
from tennis.adapters.atp_scraper.client import HttpAtpScraperClient
from tennis.adapters.http_client import HttpxClient
from tennis.adapters.odds.adapter import OddsApiAdapter
from tennis.adapters.odds.client import HttpOddsApiClient
from tennis.adapters.owm.adapter import OWMAdapter
from tennis.adapters.owm.client import HttpOWMClient
from tennis.adapters.sackmann import (
    FilesystemMirrorReader,
    SackmannAdapter,
    SackmannResolver,
    load_overrides,
)
from tennis.agents.briefing.agent import BriefingAgent
from tennis.agents.briefing.email_client import SmtpEmailClient
from tennis.agents.briefing.llm_client import AnthropicLlmClient
from tennis.agents.data.agent import DataAgent
from tennis.agents.modeling.agent import ModelingAgent
from tennis.agents.monitor.agent import MonitorAgent
from tennis.agents.orchestrator.pipeline import DailyPipeline
from tennis.agents.research.agent import ResearchAgent
from tennis.core.clock import Clock, RealClock
from tennis.core.config import AppConfig, read_required_env, validate_environment
from tennis.core.logging import get_logger
from tennis.storage.postgres import impl
from tennis.storage.postgres.session import PostgresSessionFactory

if TYPE_CHECKING:
    from uuid import UUID

    from tennis.core.contracts import Agent

_logger = get_logger("tennis.composition")

# Callable type expected by the SessionCallable contract: () -> contextmanager[Session].
SessionCallable = impl.SessionCallable


class _Wiring:
    """Builds and caches the shared infrastructure (session factory, repositories,
    HTTP transport) and constructs each agent on demand. One instance per CLI
    invocation; the agents it builds share its repositories under one process."""

    def __init__(
        self,
        config: AppConfig,
        *,
        session_factory: SessionCallable | None = None,
        validate_env: bool = True,
    ) -> None:
        self._config = config
        # §S7: fail fast on a missing required env var for the active app.env,
        # before any agent or adapter is constructed.
        if validate_env:
            validate_environment(config)

        self._owns_session = session_factory is None
        self._sf_obj: PostgresSessionFactory | None = None
        if session_factory is not None:
            self._session: SessionCallable = session_factory
        else:
            db = config.database
            self._sf_obj = PostgresSessionFactory(
                url=read_required_env(db.url_env),
                pool_size=db.pool_size,
                max_overflow=db.max_overflow,
                statement_timeout_ms=db.statement_timeout_ms,
            )
            self._session = self._sf_obj.session

        s = self._session
        self.players = impl.PlayerRepositoryImpl(s)
        self.rankings = impl.PlayerRankingRepositoryImpl(s)
        self.venues = impl.VenueRepositoryImpl(s)
        self.tournaments = impl.TournamentRepositoryImpl(s)
        self.matches = impl.MatchRepositoryImpl(s)
        self.match_stats = impl.MatchStatRepositoryImpl(s)
        self.odds = impl.OddsSnapshotRepositoryImpl(s)
        self.weather = impl.WeatherObservationRepositoryImpl(s)
        self.elo = impl.EloSnapshotRepositoryImpl(s)
        self.feature_specs = impl.FeatureSpecRepositoryImpl(s)
        self.feature_matrix = impl.FeatureMatrixRepositoryImpl(s)
        self.model_registry = impl.ModelRegistryRepositoryImpl(s)
        self.predictions = impl.PredictionRepositoryImpl(s)
        self.briefing_deliveries = impl.BriefingDeliveryRepositoryImpl(s)
        self.runs = impl.PipelineRunRepositoryImpl(s)
        self.watermarks = impl.IngestWatermarkRepositoryImpl(s)
        self.dead_letter = impl.DeadLetterRepositoryImpl(s)
        self.aliases = impl.PlayerAliasRepositoryImpl(s)

        # One shared HTTP transport for every network adapter (§S7).
        self._http = HttpxClient()

    # -- adapter factories (Callable[[Clock, UUID], Adapter]) ---------------
    def _sackmann_factory(self):
        cfg = self._config
        reader = FilesystemMirrorReader(
            cfg.sources.sackmann.local_mirror_dir, cfg.sources.sackmann.files
        )
        resolver = SackmannResolver(
            players=self.players,
            aliases=self.aliases,
            fuzzy_threshold=cfg.player_resolution.fuzzy_threshold,
            require_dob_for_fuzzy=cfg.player_resolution.require_dob_for_fuzzy,
            overrides=load_overrides(cfg.player_resolution.overrides_file),
        )

        def _make(clock: Clock, run_id: UUID) -> SackmannAdapter:
            return SackmannAdapter(
                config=cfg,
                clock=clock,
                reader=reader,
                resolver=resolver,
                players=self.players,
                rankings=self.rankings,
                tournaments=self.tournaments,
                matches=self.matches,
                match_stats=self.match_stats,
                watermarks=self.watermarks,
                dead_letter=self.dead_letter,
                run_id=run_id,
            )

        return _make

    def _scraper_factory(self):
        cfg = self._config
        client = HttpAtpScraperClient(config=cfg, http=self._http)

        def _make(clock: Clock, run_id: UUID) -> AtpScraperAdapter:
            return AtpScraperAdapter(
                config=cfg,
                clock=clock,
                client=client,
                players=self.players,
                aliases=self.aliases,
                tournaments=self.tournaments,
                matches=self.matches,
                watermarks=self.watermarks,
                dead_letter=self.dead_letter,
                run_id=run_id,
            )

        return _make

    def _odds_factory(self):
        cfg = self._config
        client = HttpOddsApiClient(
            config=cfg,
            http=self._http,
            api_key=read_required_env(cfg.sources.odds_api.api_key_env),
        )

        def _make(clock: Clock, run_id: UUID) -> OddsApiAdapter:
            return OddsApiAdapter(
                config=cfg,
                clock=clock,
                client=client,
                matches=self.matches,
                odds=self.odds,
                aliases=self.aliases,
                watermarks=self.watermarks,
                dead_letter=self.dead_letter,
                run_id=run_id,
            )

        return _make

    def _owm_factory(self):
        cfg = self._config
        client = HttpOWMClient(
            config=cfg,
            http=self._http,
            api_key=read_required_env(cfg.sources.openweather.api_key_env),
        )

        def _make(clock: Clock, run_id: UUID) -> OWMAdapter:
            return OWMAdapter(
                config=cfg,
                clock=clock,
                client=client,
                venues=self.venues,
                weather=self.weather,
                watermarks=self.watermarks,
                dead_letter=self.dead_letter,
                run_id=run_id,
            )

        return _make

    # -- agents -------------------------------------------------------------
    def data_agent(self) -> DataAgent:
        return DataAgent(
            config=self._config,
            sackmann_factory=self._sackmann_factory(),
            scraper_factory=self._scraper_factory(),
            odds_factory=self._odds_factory(),
            owm_factory=self._owm_factory(),
            venues=self.venues,
        )

    def research_agent(self, *, mode: Literal["training", "prediction"]) -> ResearchAgent:
        return ResearchAgent(
            mode=mode,
            config=self._config,
            match_repo=self.matches,
            elo_repo=self.elo,
            tournament_repo=self.tournaments,
            ranking_repo=self.rankings,
            stat_repo=self.match_stats,
            weather_repo=self.weather,
            venue_repo=self.venues,
            odds_repo=self.odds,
            feature_spec_repo=self.feature_specs,
            feature_matrix_repo=self.feature_matrix,
            dead_letter=self.dead_letter,
        )

    def modeling_agent(self, *, mode: Literal["training", "prediction"]) -> ModelingAgent:
        return ModelingAgent(
            mode=mode,
            config=self._config,
            match_repo=self.matches,
            feature_matrix_repo=self.feature_matrix,
            feature_spec_repo=self.feature_specs,
            model_registry_repo=self.model_registry,
            # Required only in prediction mode (validated in the agent ctor).
            prediction_repo=self.predictions,
            dead_letter_repo=self.dead_letter,
        )

    def briefing_agent(self) -> BriefingAgent:
        return BriefingAgent(
            config=self._config,
            model_registry_repo=self.model_registry,
            prediction_repo=self.predictions,
            match_repo=self.matches,
            player_repo=self.players,
            tournament_repo=self.tournaments,
            feature_matrix_repo=self.feature_matrix,
            dead_letter_repo=self.dead_letter,
            briefing_delivery_repo=self.briefing_deliveries,
            llm_client=AnthropicLlmClient(config=self._config.briefing.llm),
            email_client=SmtpEmailClient(config=self._config.briefing.email),
        )

    def monitor_agent(self) -> MonitorAgent:
        return MonitorAgent(
            config=self._config,
            model_registry_repo=self.model_registry,
            prediction_repo=self.predictions,
            match_repo=self.matches,
        )

    def pipeline(self, agents: list[Agent]) -> DailyPipeline:
        return DailyPipeline(
            real_clock=RealClock(),
            runs=self.runs,
            agents=agents,
            config=self._config,
            db=self._sf_obj,
        )


def build_training_chain(
    config: AppConfig,
    *,
    session_factory: SessionCallable | None = None,
    validate_env: bool = True,
) -> DailyPipeline:
    """Training chain (§S2): Data → Research(training) → Modeling(training) under
    one `run_id`. No Briefing/Monitor — training only ingests + trains + registers
    a model. Run by `python -m tennis train`, separate from the daily cron."""
    w = _Wiring(config, session_factory=session_factory, validate_env=validate_env)
    agents: list[Agent] = [
        w.data_agent(),
        w.research_agent(mode="training"),
        w.modeling_agent(mode="training"),
    ]
    return w.pipeline(agents)


def build_daily_chain(
    config: AppConfig,
    *,
    session_factory: SessionCallable | None = None,
    validate_env: bool = True,
) -> DailyPipeline:
    """Daily prediction chain (§S3): Data → Research(prediction) →
    Modeling(prediction) → Briefing → Monitor under one `run_id`. Run by
    `python -m tennis run` (06:30 UTC cron). Requires an ACTIVE model — the
    bootstrap (§S6) is to run `train` first."""
    w = _Wiring(config, session_factory=session_factory, validate_env=validate_env)
    agents: list[Agent] = [
        w.data_agent(),
        w.research_agent(mode="prediction"),
        w.modeling_agent(mode="prediction"),
        w.briefing_agent(),
    ]
    # §Q5/§Q7/A13: Monitor is post-briefing observability gated ONLY at the
    # orchestration layer by `run_monitor_post_briefing`. When disabled it is
    # omitted from the chain entirely (the agent has empty preconditions, so the
    # loop would otherwise always run it).
    if config.orchestrator.run_monitor_post_briefing:
        agents.append(w.monitor_agent())
    return w.pipeline(agents)
