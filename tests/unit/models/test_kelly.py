"""Fractional Kelly staking (M1b) — §M24 / H12."""

from __future__ import annotations

import pytest

from tennis.models.kelly import KellyStake, apply_same_day_cap, per_match_kelly


class TestPerMatch:
    def test_positive_p1_edge_sizes_p1(self, real_config):
        k = per_match_kelly(
            p1_prob_cal=0.70, p1_implied_shin=0.60,
            p1_implied_proportional=0.62, config=real_config,
        )
        assert k.kelly_fraction_p1 is not None and k.kelly_fraction_p1 > 0.0
        assert k.kelly_fraction_p2 == 0.0

    def test_per_match_cap_enforced(self, real_config):
        # huge edge → raw Kelly far exceeds cap; clamp to max_exposure_pct.
        k = per_match_kelly(
            p1_prob_cal=0.95, p1_implied_shin=0.50,
            p1_implied_proportional=None, config=real_config,
        )
        assert k.kelly_fraction_p1 == pytest.approx(
            real_config.modeling.kelly.max_exposure_pct
        )

    def test_below_threshold_is_zero(self, real_config):
        # edge 0.005 < min_edge_to_bet (0.03) → sized 0.0, not NULL (odds exist).
        k = per_match_kelly(
            p1_prob_cal=0.605, p1_implied_shin=0.60,
            p1_implied_proportional=None, config=real_config,
        )
        assert k.kelly_fraction_p1 == 0.0 and k.kelly_fraction_p2 == 0.0

    def test_no_odds_is_null(self, real_config):
        # C9: no usable implied → NULL/NULL (cannot size).
        k = per_match_kelly(
            p1_prob_cal=0.70, p1_implied_shin=None,
            p1_implied_proportional=None, config=real_config,
        )
        assert k.kelly_fraction_p1 is None and k.kelly_fraction_p2 is None

    def test_shin_fallback_proportional(self, real_config):
        # §M24c: degenerate Shin → fall back to proportional implied.
        k = per_match_kelly(
            p1_prob_cal=0.70, p1_implied_shin=1.5,  # degenerate
            p1_implied_proportional=0.60, config=real_config,
        )
        assert k.kelly_fraction_p1 is not None and k.kelly_fraction_p1 > 0.0

    def test_p2_side_sized_when_p1_underdog(self, real_config):
        k = per_match_kelly(
            p1_prob_cal=0.30, p1_implied_shin=0.40,
            p1_implied_proportional=None, config=real_config,
        )
        assert k.kelly_fraction_p2 is not None and k.kelly_fraction_p2 > 0.0
        assert k.kelly_fraction_p1 == 0.0


class TestSameDayCap:
    def test_scales_pro_rata_to_total_cap(self, real_config):
        cap = real_config.modeling.kelly.max_total_exposure_pct
        stakes = [KellyStake(0.02, 0.0)] * 8  # sum 0.16 > cap 0.10
        capped = apply_same_day_cap(stakes, config=real_config)
        total = sum(s.kelly_fraction_p1 for s in capped)
        assert total == pytest.approx(cap)

    def test_under_cap_unchanged(self, real_config):
        stakes = [KellyStake(0.01, 0.0), KellyStake(0.02, 0.0)]  # 0.03 < cap
        capped = apply_same_day_cap(stakes, config=real_config)
        assert [s.kelly_fraction_p1 for s in capped] == [0.01, 0.02]

    def test_null_preserved_through_cap(self, real_config):
        stakes = [KellyStake(0.08, 0.0), KellyStake(None, None), KellyStake(0.08, 0.0)]
        capped = apply_same_day_cap(stakes, config=real_config)
        assert capped[1].kelly_fraction_p1 is None
        total = sum((s.kelly_fraction_p1 or 0.0) for s in capped)
        assert total == pytest.approx(real_config.modeling.kelly.max_total_exposure_pct)

    def test_both_caps_apply(self, real_config):
        # H12: per-match cap THEN same-day cap. 8 maxed bets (0.02 each = 0.16)
        # → per-match already clamped, same-day scales the total to 0.10.
        stakes = [
            per_match_kelly(
                p1_prob_cal=0.95, p1_implied_shin=0.50,
                p1_implied_proportional=None, config=real_config,
            )
            for _ in range(8)
        ]
        assert all(s.kelly_fraction_p1 == 0.02 for s in stakes)  # per-match cap
        capped = apply_same_day_cap(stakes, config=real_config)
        total = sum(s.kelly_fraction_p1 for s in capped)
        assert total == pytest.approx(real_config.modeling.kelly.max_total_exposure_pct)
