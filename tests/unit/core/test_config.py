"""Config loader + environment validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tennis.core.config import (
    AppConfig,
    load_config,
    read_required_env,
    required_env_for,
    validate_environment,
)
from tennis.core.errors import InvalidConfigError, MissingEnvironmentError


@pytest.fixture
def config(config_path: Path) -> AppConfig:
    """Loaded real config — convenience for tests that don't need the path."""
    return load_config(config_path)


# --------------------------------------------------------------------------
# Loading the real config/config.yaml — guards against config drift
# --------------------------------------------------------------------------
class TestLoadRealConfig:
    def test_loads_repo_config(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        assert isinstance(cfg, AppConfig)

    def test_locked_decisions_present(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        # --- Day 1 locks ---
        assert cfg.decision_timing.live_decision_offset_hours == 24
        assert cfg.decision_timing.backtest_use_closing_line is True
        assert cfg.features.market.devig_method_primary == "shin"
        assert cfg.features.market.devig_method_fallback == "proportional"
        assert cfg.modeling.splits.embargo_by == "tournament_id"
        assert cfg.briefing.llm.model == "claude-sonnet-4-6"
        assert cfg.modeling.calibration.method == "platt"
        assert cfg.modeling.calibration.tail_days > 0

    def test_day2_review_locks_present(self, config_path: Path) -> None:
        """Day 2 architectural-review fixes — never silently regress."""
        cfg = load_config(config_path)

        # Sackmann mirror + staleness
        assert cfg.sources.sackmann.max_staleness_days == 3
        assert cfg.sources.sackmann.local_mirror_dir
        # pin_commit_sha may be null but the field must exist
        assert hasattr(cfg.sources.sackmann, "pin_commit_sha")

        # Missing-odds contract
        assert cfg.modeling.edge.allow_missing_odds is True

        # Tournament inclusion policy (centralized)
        assert "GS" in cfg.policy.tournaments.included_tiers
        assert "Masters1000" in cfg.policy.tournaments.included_tiers
        assert "ATP500" in cfg.policy.tournaments.included_tiers
        assert "ATP250" in cfg.policy.tournaments.included_tiers
        # Excluded categories listed for documentation
        for excluded in ("Davis Cup", "Olympics", "Challengers", "Futures"):
            assert excluded in cfg.policy.tournaments.excluded_categories

        # Match status filters
        assert cfg.policy.match_status.training == ("final",)
        assert set(cfg.policy.match_status.prediction) == {"scheduled", "live"}

        # Player resolution
        assert 0.0 < cfg.player_resolution.fuzzy_threshold <= 1.0
        assert cfg.player_resolution.overrides_file.endswith(".yaml")
        assert cfg.player_resolution.require_dob_for_fuzzy is True

        # Retirement / walkover semantics
        assert cfg.feature_engineering.retirement_counts_as_match is True
        assert cfg.feature_engineering.walkover_counts_as_match is False
        assert 0.0 < cfg.feature_engineering.retirement_fatigue_weight <= 1.0

        # Orchestrator heartbeat
        assert cfg.orchestrator.heartbeat.interval_s == 30
        assert cfg.orchestrator.heartbeat.orphan_after_s == 300
        assert cfg.orchestrator.orphan_sweep_on_start is True

        # Kelly disclaimer
        assert "research" in cfg.briefing.kelly_disclaimer.lower()

    def test_h_audit_fixes_present(self, config: AppConfig) -> None:
        """Regression: every H-section audit fix is locked in config."""

        # H1: noise injection is in modeling, not storage
        # (semantic — we assert inject_forecast_noise exists; the
        # architectural rule is enforced by the absence of noise logic
        # in the Research Agent, which has no tests yet)
        assert config.features.weather.inject_forecast_noise is True

        # H2: season range uses structured form, not list
        assert config.ingestion.history_backfill_season_range.start == 2000
        assert config.ingestion.history_backfill_season_range.end == 2026

        # H3: per-status canonical date source policy
        canonical = config.sources.sackmann.canonical_date_source
        assert canonical["final"] == "sackmann"
        assert canonical["scheduled"] == "atp_scraper"
        assert canonical["live"] == "atp_scraper"

        # H4: uncertainty bucket thresholds defined
        thresholds = config.features.weather.uncertainty_bucket_thresholds
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds
        assert thresholds["low"] == [0, 6]
        assert thresholds["medium"] == [6, 24]

        # H5: correct OWM endpoint
        assert "day_summary" in config.sources.openweather.historical_endpoint

        # H7: elo snapshot materialization on
        assert config.features.elo.materialize_snapshots is True

        # H8: best_of fallback defined for GS
        fallback = config.feature_engineering.best_of_null_fallback_by_tier
        assert fallback["GS"] == 5
        assert fallback["default"] == 3

        # H9: 'logit' is not an accepted devig method (constraint corrected
        # in migration 012; config Literal narrowed in Part 1).
        from pydantic import ValidationError

        from tennis.core.config import MarketConfig

        with pytest.raises(ValidationError):
            MarketConfig(devig_method_primary="logit", devig_method_fallback="shin")
        with pytest.raises(ValidationError):
            MarketConfig(devig_method_primary="shin", devig_method_fallback="logit")

        # H10: elo cold start and K-factor present
        assert config.features.elo.initial_rating == 1500
        assert config.features.elo.k_new_player > config.features.elo.k_established

        # H11: odds coverage start year documented
        assert config.sources.odds_api.coverage_start_year == 2020

        # H12: total exposure cap present and sane
        assert 0 < config.modeling.kelly.max_total_exposure_pct <= 0.20

        # Calibration floor
        assert config.modeling.calibration.min_calibration_samples >= 20


# --------------------------------------------------------------------------
# Loader error handling
# --------------------------------------------------------------------------
class TestLoadConfigErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_non_mapping_root(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(InvalidConfigError, match="mapping"):
            load_config(p)

    def test_non_utc_timezone_rejected(self, tmp_path: Path, config_path: Path) -> None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["app"]["timezone"] = "America/New_York"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(InvalidConfigError, match="UTC"):
            load_config(p)


# --------------------------------------------------------------------------
# Environment validation — the startup safety net
# --------------------------------------------------------------------------
class TestValidateEnvironment:
    def test_dev_env_only_requires_database_url(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        deps = required_env_for(cfg, env="dev")
        names = {d.name for d in deps}
        # In dev only DATABASE_URL is required; secrets are optional.
        assert "DATABASE_URL" in names
        assert "ANTHROPIC_API_KEY" not in names
        assert "OPENWEATHER_API_KEY" not in names

    def test_prod_requires_full_secret_set(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        deps = required_env_for(cfg, env="prod")
        names = {d.name for d in deps}
        expected = {
            "DATABASE_URL",
            "QDRANT_URL",
            "VOYAGE_API_KEY",
            "OPENWEATHER_API_KEY",
            "ODDS_API_KEY",
            "ANTHROPIC_API_KEY",
            "SMTP_HOST",
            "SMTP_USER",
            "SMTP_PASSWORD",
            "BRIEFING_RECIPIENTS",
        }
        missing = expected - names
        assert missing == set(), f"prod env deps missing: {missing}"

    def test_passes_when_all_set(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        env_pairs = tuple(
            (d.name, "x") for d in required_env_for(cfg, env="prod")
        )
        validate_environment(cfg, env="prod", environ=env_pairs)

    def test_fails_loudly_with_all_missing(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        with pytest.raises(MissingEnvironmentError) as excinfo:
            validate_environment(cfg, env="prod", environ=())
        msg = str(excinfo.value)
        # Every required var name should appear in the error message
        for dep in required_env_for(cfg, env="prod"):
            assert dep.name in msg
        assert "prod" in msg

    def test_treats_blank_string_as_missing(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        env_pairs = tuple(
            (d.name, "" if d.name == "ANTHROPIC_API_KEY" else "x")
            for d in required_env_for(cfg, env="prod")
        )
        with pytest.raises(MissingEnvironmentError) as excinfo:
            validate_environment(cfg, env="prod", environ=env_pairs)
        assert "ANTHROPIC_API_KEY" in str(excinfo.value)


class TestReadRequiredEnv:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_KEY", "value-123")
        assert read_required_env("SOME_KEY") == "value-123"

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_KEY", raising=False)
        with pytest.raises(MissingEnvironmentError):
            read_required_env("SOME_KEY")

    def test_raises_when_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_KEY", "   ")
        with pytest.raises(MissingEnvironmentError):
            read_required_env("SOME_KEY")
