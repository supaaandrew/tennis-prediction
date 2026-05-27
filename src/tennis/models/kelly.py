"""Fractional Kelly staking with per-match + same-day caps (§M24 / H12).

Sizing basis (§M24c): the Shin decision-implied prob, falling back to the
proportional implied when Shin is NULL/degenerate (mirrors
`config.features.market.devig_method_primary→fallback`). Using the devigged
implied as the fair estimate, the full-Kelly fraction for the favoured side is
`edge / (1 − implied_side)`; it is scaled by `config.modeling.kelly.fraction`.

Both caps ALWAYS apply (H12):
  - **per-match** — each stake clamped to `kelly.max_exposure_pct`;
  - **same-day** (§M24a/b) — keyed on `predicted_at.date()`; if the day's stakes
    sum above `kelly.max_total_exposure_pct`, every stake is scaled pro-rata so
    the day's total equals the cap.

No-bet semantics (§M24d): `kelly_fraction_*` is NULL when no usable odds exist
(C9) and 0.0 when odds exist but `edge < config.modeling.edge.min_edge_to_bet`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tennis.core.config import AppConfig


@dataclass(frozen=True, slots=True)
class KellyStake:
    """Per-match Kelly fractions. NULL ⟺ no usable odds; 0.0 ⟺ no qualifying edge."""

    kelly_fraction_p1: float | None
    kelly_fraction_p2: float | None


def _pick_implied(shin: float | None, proportional: float | None) -> float | None:
    """Shin first, proportional fallback (§M24c). Both must be a usable prob."""
    if shin is not None and 0.0 < shin < 1.0:
        return shin
    if proportional is not None and 0.0 < proportional < 1.0:
        return proportional
    return None


def per_match_kelly(
    *,
    p1_prob_cal: float,
    p1_implied_shin: float | None,
    p1_implied_proportional: float | None,
    config: AppConfig,
) -> KellyStake:
    """Per-match fractional Kelly (pre same-day cap).

    Returns NULL/NULL when no usable implied prob exists (C9). Otherwise exactly
    one side can carry a positive edge (two-outcome market), so at most one
    fraction is positive; the other is 0.0.
    """
    implied_p1 = _pick_implied(p1_implied_shin, p1_implied_proportional)
    if implied_p1 is None:
        return KellyStake(None, None)

    frac = config.modeling.kelly.fraction
    cap = config.modeling.kelly.max_exposure_pct
    min_edge = config.modeling.edge.min_edge_to_bet

    edge_p1 = p1_prob_cal - implied_p1
    edge_p2 = -edge_p1  # devigged two-outcome market sums to 1
    k1 = k2 = 0.0
    if edge_p1 >= min_edge:
        # denom = 1 - implied_p1 (favoured side p1)
        k1 = min(frac * edge_p1 / (1.0 - implied_p1), cap)
    elif edge_p2 >= min_edge:
        # implied_p2 = 1 - implied_p1, so denom = 1 - implied_p2 = implied_p1
        k2 = min(frac * edge_p2 / implied_p1, cap)
    return KellyStake(kelly_fraction_p1=k1, kelly_fraction_p2=k2)


def apply_same_day_cap(
    stakes: Sequence[KellyStake], *, config: AppConfig
) -> list[KellyStake]:
    """Scale a single day's stakes pro-rata to the same-day total cap (§M24a/b).

    `stakes` are all the bets sharing one `predicted_at.date()`. NULLs pass
    through unchanged; 0.0 stays 0.0. No scaling when the day's total is already
    within the cap.
    """
    cap = config.modeling.kelly.max_total_exposure_pct
    total = sum(
        (s.kelly_fraction_p1 or 0.0) + (s.kelly_fraction_p2 or 0.0) for s in stakes
    )
    if total <= cap or total == 0.0:
        return list(stakes)
    scale = cap / total
    return [
        KellyStake(
            kelly_fraction_p1=(
                None if s.kelly_fraction_p1 is None else s.kelly_fraction_p1 * scale
            ),
            kelly_fraction_p2=(
                None if s.kelly_fraction_p2 is None else s.kelly_fraction_p2 * scale
            ),
        )
        for s in stakes
    ]
