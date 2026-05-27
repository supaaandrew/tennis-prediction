"""Edge computation vs implied probs (M1b) — §M21a + C9."""

from __future__ import annotations

import pytest

from tennis.models.edge import compute_edges


class TestEdges:
    def test_shin_and_proportional_edges(self):
        er = compute_edges(
            0.70,
            {
                "p1_implied_pinnacle_decision": 0.60,
                "p1_implied_proportional_decision": 0.62,
            },
        )
        assert er.edge_p1_shin == pytest.approx(0.10)
        assert er.edge_p1_proportional == pytest.approx(0.08)

    def test_p2_edge_mirrors_p1(self):
        # Two-outcome devigged market sums to 1 → edge_p2 == -edge_p1.
        er = compute_edges(0.70, {"p1_implied_pinnacle_decision": 0.60})
        assert er.edge_p2_shin == pytest.approx(-er.edge_p1_shin)

    def test_implied_passthrough(self):
        er = compute_edges(0.5, {"p1_implied_pinnacle_decision": 0.55})
        assert er.p1_implied_shin == pytest.approx(0.55)


class TestMissingOdds:
    def test_missing_pinnacle_nulls_shin_edges(self):
        # C9: no Pinnacle implied → shin edges NULL (prediction still written).
        er = compute_edges(0.70, {"p1_implied_proportional_decision": 0.62})
        assert er.edge_p1_shin is None and er.edge_p2_shin is None
        assert er.edge_p1_proportional is not None

    def test_empty_payload_all_edges_null(self):
        er = compute_edges(0.70, {})
        assert er.edge_p1_shin is None
        assert er.edge_p1_proportional is None
        assert er.p1_implied_shin is None

    def test_out_of_range_implied_treated_as_missing(self):
        er = compute_edges(0.70, {"p1_implied_pinnacle_decision": 1.4})
        assert er.p1_implied_shin is None and er.edge_p1_shin is None
