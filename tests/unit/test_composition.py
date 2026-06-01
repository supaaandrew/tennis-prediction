"""Composition-root tests (§S7).

The builders construct the full agent graph from config + env WITHOUT touching
the network or a real DB: adapter factories are closures (no egress at build),
the LLM/SMTP clients only store credentials at construction, and the session
factory is injected. We assert the right agent set + modes per command and that a
missing required secret fails loudly.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tennis.composition import build_daily_chain, build_training_chain
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import MissingEnvironmentError


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "config" / "config.yaml")


def _fake_session_factory() -> Any:
    """A SessionCallable whose context manager is never entered at construction
    (repositories only store it). Returns a no-op session if ever called."""

    @contextmanager
    def _cm():
        yield MagicMock()

    return _cm


def _set_all_secrets(monkeypatch: pytest.MonkeyPatch, config: AppConfig) -> None:
    monkeypatch.setenv(config.database.url_env, "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv(config.sources.odds_api.api_key_env, "dummy-odds")
    monkeypatch.setenv(config.sources.openweather.api_key_env, "dummy-owm")
    monkeypatch.setenv(config.sources.matchstat.api_key_env, "dummy-matchstat")
    monkeypatch.setenv(config.briefing.llm.api_key_env, "dummy-anthropic")
    monkeypatch.setenv(config.briefing.email.smtp.host_env, "smtp.example.com")
    monkeypatch.setenv(config.briefing.email.smtp.user_env, "mailer")
    monkeypatch.setenv(config.briefing.email.smtp.password_env, "pw")
    monkeypatch.setenv(config.briefing.email.recipients_env, "a@example.com")


def _names(pipeline: Any) -> list[str]:
    return [a.name for a in pipeline._agents]


class TestBuildTrainingChain:
    def test_agent_set_and_modes(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_all_secrets(monkeypatch, config)
        pipeline = build_training_chain(config, session_factory=_fake_session_factory())
        assert _names(pipeline) == ["data", "research", "modeling"]
        # mode flags: training-mode Research + Modeling (no Briefing/Monitor).
        research, modeling = pipeline._agents[1], pipeline._agents[2]
        assert research._mode == "training"
        assert modeling._mode == "training"
        # §S6a: training tolerates a blocked scraper (Sackmann history only).
        assert pipeline._agents[0]._scraper_required is False


class TestBuildDailyChain:
    def test_agent_set_and_modes(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_all_secrets(monkeypatch, config)
        pipeline = build_daily_chain(config, session_factory=_fake_session_factory())
        # default config has run_monitor_post_briefing=True → Monitor included.
        assert _names(pipeline) == ["data", "research", "modeling", "briefing", "monitor"]
        research, modeling = pipeline._agents[1], pipeline._agents[2]
        assert research._mode == "prediction"
        assert modeling._mode == "prediction"
        # §S6a: the daily run REQUIRES the scraper (it supplies the live slate).
        assert pipeline._agents[0]._scraper_required is True

    def test_monitor_omitted_when_run_monitor_flag_false(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # §Q5/§Q7/A13: run_monitor_post_briefing=False drops Monitor entirely.
        _set_all_secrets(monkeypatch, config)
        cfg_off = config.model_copy(
            update={
                "orchestrator": config.orchestrator.model_copy(
                    update={"run_monitor_post_briefing": False}
                )
            }
        )
        pipeline = build_daily_chain(cfg_off, session_factory=_fake_session_factory())
        names = _names(pipeline)
        assert names == ["data", "research", "modeling", "briefing"]
        assert "monitor" not in names


class TestT1MatchstatWiring:
    """§T1 — the scraper-factory slot now builds `MatchstatScraperAdapter`,
    not the deprecated `HttpAtpScraperClient`. The daily chain in prod
    requires `MATCHSTAT_API_KEY`; training never does (§T1 extension of §S6a)."""

    def test_scraper_factory_builds_matchstat_adapter(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from tennis.adapters.matchstat.adapter import MatchstatScraperAdapter
        from tennis.core.clock import FrozenClock

        _set_all_secrets(monkeypatch, config)
        pipeline = build_daily_chain(config, session_factory=_fake_session_factory())
        data_agent = pipeline._agents[0]
        clock = FrozenClock(datetime(2026, 5, 30, tzinfo=UTC))
        scraper = data_agent._scraper_factory(clock, uuid4())
        assert isinstance(scraper, MatchstatScraperAdapter)

    def test_daily_in_prod_requires_matchstat_key(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prod = config.model_copy(
            update={"app": config.app.model_copy(update={"env": "prod"})}
        )
        _set_all_secrets(monkeypatch, prod)
        monkeypatch.delenv(prod.sources.matchstat.api_key_env, raising=False)
        with pytest.raises(MissingEnvironmentError):
            build_daily_chain(prod, session_factory=_fake_session_factory())


class TestSecretValidation:
    def test_missing_database_url_raises(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DATABASE_URL is required even in dev; validate_environment fails fast.
        monkeypatch.delenv(config.database.url_env, raising=False)
        with pytest.raises(MissingEnvironmentError):
            build_daily_chain(config)

    def test_missing_adapter_api_key_raises(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All secrets except the odds key — construction of the odds client reads
        # it via read_required_env and fails loudly.
        _set_all_secrets(monkeypatch, config)
        monkeypatch.delenv(config.sources.odds_api.api_key_env, raising=False)
        with pytest.raises(MissingEnvironmentError):
            build_daily_chain(config, session_factory=_fake_session_factory())


class TestTrainingScopedValidation:
    """§S6a: training validates only DB + source-adapter keys, never the
    briefing-only secrets it does not construct — even in prod."""

    def test_training_omits_briefing_secrets_in_prod_but_daily_requires_them(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prod = config.model_copy(
            update={"app": config.app.model_copy(update={"env": "prod"})}
        )
        # Only DB + the two source-adapter keys are present.
        monkeypatch.setenv(prod.database.url_env, "postgresql+psycopg://u:p@h:5432/db")
        monkeypatch.setenv(prod.sources.odds_api.api_key_env, "odds")
        monkeypatch.setenv(prod.sources.openweather.api_key_env, "owm")
        # The briefing-only secrets are absent. The matchstat key is ALSO
        # absent — §T1 extension of §S6a: training never builds the matchstat
        # slate adapter, so its key is filtered out of training validation
        # deps and a training backfill works on a machine without it.
        for name in (
            prod.briefing.llm.api_key_env,
            prod.briefing.email.smtp.host_env,
            prod.briefing.email.smtp.user_env,
            prod.briefing.email.smtp.password_env,
            prod.briefing.email.recipients_env,
            prod.sources.matchstat.api_key_env,
        ):
            monkeypatch.delenv(name, raising=False)

        # Training builds without raising — it never needs the briefing secrets.
        pipeline = build_training_chain(prod, session_factory=_fake_session_factory())
        assert _names(pipeline) == ["data", "research", "modeling"]

        # The daily chain in prod DOES require them → fails loudly.
        with pytest.raises(MissingEnvironmentError):
            build_daily_chain(prod, session_factory=_fake_session_factory())


class TestT2SidecarConsumers:
    """§T-2 — composition wires all four sidecar consumers into the agent
    graph when the matchstat key is configured (daily chain), and skips
    them silently when absent (training chain)."""

    def test_daily_chain_threads_all_four_consumers(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tennis.adapters.matchstat.client import MatchstatClient
        from tennis.adapters.matchstat.sidecars import MatchstatH2hStatsClient

        _set_all_secrets(monkeypatch, config)
        pipeline = build_daily_chain(config, session_factory=_fake_session_factory())
        data, research, _modeling, _briefing, monitor = pipeline._agents
        # §T12 / §T13 — DataAgent receives both sidecar factories.
        assert data._matchstat_calendar_factory is not None
        assert data._matchstat_rankings_factory is not None
        # §T5 — pre-fetch check wired to MatchstatClient.verify_tier_ids.
        assert data._pre_fetch_check is not None
        # §T14 — ResearchAgent receives the shared h2h-stats client.
        assert isinstance(research._h2h_stats_client, MatchstatH2hStatsClient)
        # §T6 — Monitor receives the SAME singleton matchstat client.
        assert isinstance(monitor._matchstat_client, MatchstatClient)

    def test_training_chain_skips_matchstat_consumers(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training has no key; the four consumers are all None / unwired.
        Each absent consumer is documented degrade-safe at its site."""
        prod = config.model_copy(
            update={"app": config.app.model_copy(update={"env": "prod"})}
        )
        # Only DB + the two source-adapter keys are present (and NO matchstat key).
        monkeypatch.setenv(prod.database.url_env, "postgresql+psycopg://u:p@h:5432/db")
        monkeypatch.setenv(prod.sources.odds_api.api_key_env, "odds")
        monkeypatch.setenv(prod.sources.openweather.api_key_env, "owm")
        for name in (
            prod.briefing.llm.api_key_env,
            prod.briefing.email.smtp.host_env,
            prod.briefing.email.smtp.user_env,
            prod.briefing.email.smtp.password_env,
            prod.briefing.email.recipients_env,
            prod.sources.matchstat.api_key_env,
        ):
            monkeypatch.delenv(name, raising=False)

        pipeline = build_training_chain(prod, session_factory=_fake_session_factory())
        data, research, _modeling = pipeline._agents
        # All four sidecar surfaces are None / unwired.
        assert data._matchstat_calendar_factory is None
        assert data._matchstat_rankings_factory is None
        assert data._pre_fetch_check is None
        assert research._h2h_stats_client is None
