"""Tail calibration (M1b) — §M21c + degraded path (pre-step 5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from tennis.core.config import AppConfig
from tennis.models.calibration import fit_calibrator


def _tail(n: int = 200, *, seed: int = 1):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < raw).astype(int)  # raw ≈ calibrated truth
    return raw, y


def _min_samples(config: AppConfig, n: int) -> AppConfig:
    return config.model_copy(
        update={
            "modeling": config.modeling.model_copy(
                update={
                    "calibration": config.modeling.calibration.model_copy(
                        update={"min_calibration_samples": n}
                    )
                }
            )
        }
    )


class TestFit:
    def test_platt_maps_to_unit_interval(self, real_config):
        raw, y = _tail()
        cal = fit_calibrator(raw, y, config=_min_samples(real_config, 50))
        assert cal.degraded is False
        out = cal.predict(raw)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_isotonic_method(self, real_config):
        raw, y = _tail()
        cfg = real_config.model_copy(
            update={
                "modeling": real_config.modeling.model_copy(
                    update={
                        "calibration": real_config.modeling.calibration.model_copy(
                            update={"method": "isotonic", "min_calibration_samples": 50}
                        )
                    }
                )
            }
        )
        cal = fit_calibrator(raw, y, config=cfg)
        assert cal.method == "isotonic" and cal.degraded is False


class TestDegraded:
    def test_below_min_samples_degraded_passthrough(self, real_config):
        raw, y = _tail(n=10)
        cal = fit_calibrator(raw, y, config=_min_samples(real_config, 50))
        assert cal.degraded is True and cal.model is None
        # passthrough clips only — identity on in-range inputs.
        np.testing.assert_allclose(cal.predict(raw), np.clip(raw, 0.0, 1.0))

    def test_empty_tail_degraded(self, real_config):
        # pre-step 5.1: len==0 tail behaves like < min_calibration_samples.
        cal = fit_calibrator(np.asarray([]), np.asarray([]), config=real_config)
        assert cal.degraded is True and cal.n_samples == 0

    def test_single_class_tail_degraded(self, real_config):
        raw = np.linspace(0.1, 0.9, 100)
        cal = fit_calibrator(raw, np.ones(100, dtype=int), config=_min_samples(real_config, 50))
        assert cal.degraded is True  # cannot calibrate one class

    def test_passthrough_clips_out_of_range(self, real_config):
        cal = fit_calibrator(np.asarray([]), np.asarray([]), config=real_config)
        out = cal.predict(np.asarray([-0.3, 0.4, 1.4]))
        assert out.tolist() == [0.0, 0.4, 1.0]
