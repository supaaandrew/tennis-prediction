"""Secondary modeling metrics (M1b): ECE + Kelly-ROI backtest."""

from __future__ import annotations

import numpy as np

from tennis.models.metrics import (
    expected_calibration_error,
    population_stability_index,
    roi_kelly_backtest,
)


class TestECE:
    def test_perfectly_calibrated_low_ece(self):
        # predictions == empirical accuracy per bin → ECE ≈ 0.
        rng = np.random.default_rng(0)
        p = rng.uniform(0.0, 1.0, size=5000)
        y = (rng.uniform(0.0, 1.0, size=5000) < p).astype(int)
        assert expected_calibration_error(y, p) < 0.05

    def test_miscalibrated_high_ece(self):
        # always predict 0.99 but outcomes are 50/50 → large ECE.
        p = np.full(1000, 0.99)
        y = np.tile([0, 1], 500)
        assert expected_calibration_error(y, p) > 0.4

    def test_empty_is_nan(self):
        assert np.isnan(expected_calibration_error(np.asarray([]), np.asarray([])))


class TestPSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.uniform(0.0, 1.0, size=5000)
        # Same distribution (independent draws) → PSI ≈ 0.
        other = rng.uniform(0.0, 1.0, size=5000)
        assert population_stability_index(sample, other) < 0.05

    def test_shifted_distribution_high_psi(self):
        # Reference concentrated low, actual concentrated high → large PSI.
        rng = np.random.default_rng(1)
        expected = rng.uniform(0.0, 0.3, size=2000)
        actual = rng.uniform(0.7, 1.0, size=2000)
        assert population_stability_index(actual, expected) > 0.2

    def test_empty_actual_is_nan(self):
        assert np.isnan(population_stability_index(np.asarray([]), np.asarray([0.5])))

    def test_empty_expected_is_nan(self):
        # First-ever run: the reference window has no rows → NaN (Monitor → None).
        assert np.isnan(population_stability_index(np.asarray([0.5]), np.asarray([])))

    def test_empty_bin_does_not_blow_up(self):
        # An empty bin on one side must contribute a FINITE term (epsilon floor),
        # never ±inf/NaN.
        psi = population_stability_index(
            np.asarray([0.05, 0.05, 0.05]), np.asarray([0.95, 0.95, 0.95])
        )
        assert np.isfinite(psi)


class TestRoiKelly:
    def test_edge_in_our_favour_positive_roi(self, real_config):
        # model always right and edged vs implied → strategy profits.
        probs = [0.9] * 100
        y = [1] * 100
        implied = [0.6] * 100
        roi = roi_kelly_backtest(
            probs=probs, y_true=y, implied_p1=implied, config=real_config
        )
        assert roi > 0.0

    def test_no_usable_odds_zero_roi(self, real_config):
        roi = roi_kelly_backtest(
            probs=[0.7] * 10, y_true=[1] * 10, implied_p1=[None] * 10,
            config=real_config,
        )
        assert roi == 0.0

    def test_no_qualifying_edge_zero_roi(self, real_config):
        # edge 0.005 < min_edge_to_bet → nothing staked → ROI 0.
        roi = roi_kelly_backtest(
            probs=[0.605] * 10, y_true=[1] * 10, implied_p1=[0.60] * 10,
            config=real_config,
        )
        assert roi == 0.0
