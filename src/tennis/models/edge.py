"""Edge computation vs bookmaker implied probabilities (M1b).

`edge = p1_prob_cal − p1_implied_*`, computed under BOTH the Shin and the
proportional devigs, reading the implied probs **straight from the
`feature_matrix` payload** (`p1_implied_pinnacle_decision` = Shin,
`p1_implied_proportional_decision`). The model never saw the market family (§M21a
edge-circularity guard), so the edge is a meaningful diff.

Two-outcome devigged markets sum to 1, so the p2 side mirrors p1 exactly:
`edge_p2 = (1 − p1_prob_cal) − (1 − implied_p1) = −edge_p1`. Computed explicitly
for readability.

C9 / H11: when an implied prob is absent (no Pinnacle / proportional snapshot at
decision time), the corresponding edges are NULL — the prediction row is still
written (never rejected).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Market payload keys (§M19 / §M21a) — read for the edge step only, never as X.
_KEY_SHIN = "p1_implied_pinnacle_decision"
_KEY_PROPORTIONAL = "p1_implied_proportional_decision"


@dataclass(frozen=True, slots=True)
class EdgeResult:
    """The decision-time implied probs + p1/p2 edges under both devigs."""

    p1_implied_shin: float | None
    p1_implied_proportional: float | None
    edge_p1_shin: float | None
    edge_p2_shin: float | None
    edge_p1_proportional: float | None
    edge_p2_proportional: float | None


def _as_prob(value: object) -> float | None:
    """Coerce a payload value to a probability in the open interval (0, 1).

    A missing (None) or out-of-range value → None (treated as no usable odds, C9),
    so the edge against it is NULL rather than a nonsensical number.
    """
    if value is None:
        return None
    p = float(value)  # type: ignore[arg-type]
    return p if 0.0 < p < 1.0 else None


def compute_edges(p1_prob_cal: float, payload: Mapping[str, object]) -> EdgeResult:
    """Edges of the calibrated p1 probability vs the payload's implied probs."""
    shin = _as_prob(payload.get(_KEY_SHIN))
    prop = _as_prob(payload.get(_KEY_PROPORTIONAL))
    return EdgeResult(
        p1_implied_shin=shin,
        p1_implied_proportional=prop,
        edge_p1_shin=None if shin is None else p1_prob_cal - shin,
        edge_p2_shin=None if shin is None else (1.0 - p1_prob_cal) - (1.0 - shin),
        edge_p1_proportional=None if prop is None else p1_prob_cal - prop,
        edge_p2_proportional=(
            None if prop is None else (1.0 - p1_prob_cal) - (1.0 - prop)
        ),
    )
