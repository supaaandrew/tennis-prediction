"""H1 forecast-noise injection (§M22)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tennis.core.config import AppConfig
from tennis.models.noise import apply_noise

_WEATHER = "temp_c_decision"


def _offset(config: AppConfig, hours: int) -> AppConfig:
    return config.model_copy(
        update={
            "decision_timing": config.decision_timing.model_copy(
                update={"live_decision_offset_hours": hours}
            )
        }
    )


def _noise_off(config: AppConfig) -> AppConfig:
    return config.model_copy(
        update={
            "features": config.features.model_copy(
                update={
                    "weather": config.features.weather.model_copy(
                        update={"inject_forecast_noise": False}
                    )
                }
            )
        }
    )


def _frame(n: int = 4000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            _WEATHER: np.zeros(n),
            "wind_dir_deg_decision": np.full(n, 180.0),  # no sigma sub-key
            "elo_diff": np.arange(n, dtype="float64"),    # non-weather
        }
    )


class TestSigmaSelection:
    def test_uses_decision_offset_high_bucket(self, real_config):
        # 24h offset → "high" bucket (§M15: h>=24). temp_c high sigma = 3.0.
        out = apply_noise(_frame(), _offset(real_config, 24))
        assert abs(out[_WEATHER].std() - 3.0) < 0.25  # empirical ≈ high sigma

    def test_low_bucket_smaller_sigma(self, real_config):
        # 3h offset → "low" bucket. temp_c low sigma = 0.5 ≪ high sigma.
        out = apply_noise(_frame(), _offset(real_config, 3))
        assert abs(out[_WEATHER].std() - 0.5) < 0.1


class TestScope:
    def test_non_weather_columns_untouched(self, real_config):
        X = _frame()
        out = apply_noise(X, real_config)
        assert out["elo_diff"].tolist() == X["elo_diff"].tolist()

    def test_column_without_sigma_key_untouched(self, real_config):
        X = _frame()
        out = apply_noise(X, real_config)
        # wind_dir_deg_decision has no noise_sigma sub-key → left as-is.
        assert out["wind_dir_deg_decision"].tolist() == X["wind_dir_deg_decision"].tolist()

    def test_nan_preserved(self, real_config):
        X = pd.DataFrame({_WEATHER: [10.0, np.nan, 20.0]})
        out = apply_noise(X, real_config)
        assert bool(np.isnan(out[_WEATHER].iloc[1]))

    def test_returns_copy_not_mutating_input(self, real_config):
        X = _frame(50)
        snapshot = X.copy()
        out = apply_noise(X, real_config)
        assert out is not X
        pd.testing.assert_frame_equal(X, snapshot)  # input untouched (H1)


class TestDeterminismAndToggle:
    def test_seeded_deterministic(self, real_config):
        X = _frame(100)
        a = apply_noise(X, real_config)
        b = apply_noise(X, real_config)
        pd.testing.assert_frame_equal(a, b)  # seeded RNG → reproducible

    def test_inject_false_is_noop(self, real_config):
        X = _frame(100)
        out = apply_noise(X, _noise_off(real_config))
        pd.testing.assert_frame_equal(out, X)  # honored no-op (§M21 weather-skew risk)
