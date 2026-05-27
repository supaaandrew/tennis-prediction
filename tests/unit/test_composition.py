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
