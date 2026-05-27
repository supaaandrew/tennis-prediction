"""Pure rendering tests for the Briefing Agent (B1, `render.py`).

Regressions for the locked render decisions: C13 disclaimer on EVERY Kelly line,
C9 NULL-edge → "no market", §M24 Kelly NULL-vs-0.0 distinction, §M19/§15.4
closing fields never rendered, `subject_template` fill, narrative-degraded body.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tennis.agents.briefing.render import SurfacedMatch, render_email

_NOW = datetime(2026, 5, 26, 6, 30, tzinfo=UTC)
_DISC = "RESEARCH ONLY — NOT ADVICE"


def _sm(**kw) -> SurfacedMatch:
    base = dict(
        match_id=1,
        p1_name="Alice",
        p2_name="Bob",
        tournament="Test Open",
        surface="Hard",
        round="QF",
        p1_prob_cal=0.65,
        edge_p1=0.05,
        edge_p2=-0.04,
        edge_method="shin",
        kelly_fraction_p1=0.02,
        kelly_fraction_p2=0.0,
        no_market=False,
    )
    base.update(kw)
    return SurfacedMatch(**base)


def _render(matches, *, narrative="A narrative.", disclaimer=_DISC):
    return render_email(
        matches=matches,
        narrative=narrative,
        kelly_disclaimer=disclaimer,
        subject_template="Tennis edges — {date}",
        as_of=_NOW,
        model_version="2016-2019-v1",
    )


class TestSubject:
    def test_subject_template_filled_with_date(self):
        subject, _ = _render([_sm()])
        assert subject == "Tennis edges — 2026-05-26"


class TestKellyDisclaimerC13:
    def test_disclaimer_on_every_kelly_line_plus_footer(self):
        # 3 matches → 3 Kelly lines each carrying the disclaimer + 1 footer.
        matches = [_sm(match_id=1), _sm(match_id=2, no_market=True), _sm(match_id=3)]
        _, body = _render(matches)
        assert body.count(_DISC) == len(matches) + 1

    def test_no_market_kelly_line_also_carries_disclaimer(self):
        _, body = _render([_sm(no_market=True)])
        # the single match's Kelly line + the footer
        assert body.count(_DISC) == 2


class TestNoMarketC9:
    def test_null_edge_renders_no_market_not_zero(self):
        _, body = _render([_sm(no_market=True, edge_p1=None, edge_p2=None,
                               kelly_fraction_p1=None, kelly_fraction_p2=None)])
        assert "no market" in body
        assert "+0.0%" not in body


class TestKellyDistinctionM24:
    def test_none_zero_and_positive_render_distinctly(self):
        matches = [
            _sm(match_id=1, kelly_fraction_p1=None, kelly_fraction_p2=None),
            _sm(match_id=2, kelly_fraction_p1=0.0, kelly_fraction_p2=0.0),
            _sm(match_id=3, kelly_fraction_p1=0.05, kelly_fraction_p2=0.0),
        ]
        _, body = _render(matches)
        assert "couldn't price" in body          # None → no usable odds
        assert "no bet (edge below threshold)" in body  # 0.0 → priced, passed
        assert "5.00% of bankroll" in body        # >0 → capped stake


class TestClosingNeverRenderedM19:
    def test_no_closing_or_drift_fields_in_body(self):
        _, body = _render([_sm()])
        assert "closing" not in body.lower()
        assert "drift" not in body.lower()


class TestNarrative:
    def test_narrative_included_when_present(self):
        _, body = _render([_sm()], narrative="UNIQUE_NARRATIVE_MARKER")
        assert "UNIQUE_NARRATIVE_MARKER" in body

    def test_degraded_body_when_narrative_none(self):
        _, body = _render([_sm()], narrative=None)
        assert "Narrative unavailable" in body
        # the edge/Kelly content is still present
        assert "Alice" in body and _DISC in body
