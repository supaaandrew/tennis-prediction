"""Pure email rendering for the Briefing Agent (B1).

Zero I/O, no repos, no network — `render_email` turns a list of already-classified
`SurfacedMatch` rows plus an optional Claude narrative into a `(subject, body)`
pair. Every rendering rule that a locked decision touches lives here so it is
unit-testable in isolation:

  - **C9 / §M19** — a row with no market (`no_market=True`) renders as
    ``"no market"``, never as a zero edge and never dropped.
  - **C13** — the `kelly_disclaimer` is rendered on EVERY line that shows a Kelly
    fraction (the ethical guard travels with the number, never once at the top).
  - **§M24** — the Kelly figures are displayed as given (already per-match +
    same-day capped by the Modeling Agent); the NULL (no odds → couldn't price)
    vs 0.0 (priced, edge below `min_edge_to_bet` → no bet) distinction is shown
    honestly.
  - **§M19 / §15.4** — closing-line fields are NOT carried on `SurfacedMatch` at
    all, so they can never reach the email (a structural guarantee, not a filter).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

# The devig method whose edge is surfaced as the headline figure. Pinned (not a
# config knob) — Shin is the project's primary devig (§M24); the agent falls back
# to the proportional edge per-row and records which one it used here.
_PRIMARY_METHOD = "shin"
_FALLBACK_METHOD = "proportional"
_NO_MARKET = "no market"


@dataclass(frozen=True, slots=True)
class SurfacedMatch:
    """One surfaced prediction, fully resolved for display.

    Carries ONLY decision-time facts — there is deliberately no
    `p1_implied_close` / `odds_drift_to_close` field, so the backtest-only
    closing line (§M19/§15.4, NULL in every live row) cannot be rendered.

    `edge_p1`/`edge_p2` are the primary-method edge per side (Shin, or the
    proportional fallback when Shin was NULL); both are None on a no-market row.
    `kelly_fraction_*` follow §M24: None = no usable odds (couldn't price),
    0.0 = priced but edge below `min_edge_to_bet` (no bet), >0 = the capped stake.
    """

    match_id: int
    p1_name: str
    p2_name: str
    tournament: str
    surface: str
    round: str
    p1_prob_cal: float
    edge_p1: float | None
    edge_p2: float | None
    edge_method: str  # "shin" | "proportional" | "none"
    kelly_fraction_p1: float | None
    kelly_fraction_p2: float | None
    no_market: bool


def render_email(
    *,
    matches: Sequence[SurfacedMatch],
    narrative: str | None,
    kelly_disclaimer: str,
    subject_template: str,
    as_of: datetime,
    model_version: str,
) -> tuple[str, str]:
    """Render the briefing into `(subject, body)`.

    `narrative=None` is the LLM-degraded path (§N4): the prose section is
    replaced with a one-line note and the email STILL renders + sends. The
    edge/Kelly table is the load-bearing content either way.
    """
    date_str = as_of.date().isoformat()
    subject = subject_template.format(date=date_str)

    lines: list[str] = []
    lines.append(f"Tennis edge briefing — {date_str}")
    lines.append(f"Model: {model_version}")
    lines.append("")

    if narrative:
        lines.append(narrative.strip())
    else:
        lines.append("(Narrative unavailable this run — see edges below.)")
    lines.append("")

    lines.append(f"Surfaced matches ({len(matches)}):")
    lines.append("")
    for m in matches:
        lines.extend(_render_match(m, kelly_disclaimer))
        lines.append("")

    lines.append(kelly_disclaimer)
    body = "\n".join(lines).rstrip() + "\n"
    return subject, body


def _render_match(m: SurfacedMatch, kelly_disclaimer: str) -> list[str]:
    """Render one match block. The Kelly line ALWAYS carries the C13 disclaimer."""
    p2_prob = 1.0 - m.p1_prob_cal
    out = [
        f"{m.p1_name} vs {m.p2_name} — {m.tournament} ({m.surface}), {m.round}",
        f"  Model: {m.p1_name} {_pct(m.p1_prob_cal)} / {m.p2_name} {_pct(p2_prob)}",
    ]
    if m.no_market:
        out.append(f"  Edge: {_NO_MARKET}")
        out.append(
            f"  Kelly: {_NO_MARKET} (no odds — couldn't price)  [{kelly_disclaimer}]"
        )
        return out

    method = m.edge_method if m.edge_method != "none" else _PRIMARY_METHOD
    out.append(
        f"  Edge ({method}): {m.p1_name} {_edge(m.edge_p1)}, "
        f"{m.p2_name} {_edge(m.edge_p2)}"
    )
    out.append(
        f"  Kelly: {m.p1_name} {_kelly(m.kelly_fraction_p1)}, "
        f"{m.p2_name} {_kelly(m.kelly_fraction_p2)}  [{kelly_disclaimer}]"
    )
    return out


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _edge(edge: float | None) -> str:
    """Edge as a signed percent; None → 'no market' (C9 — never a zero edge)."""
    if edge is None:
        return _NO_MARKET
    return f"{edge * 100:+.1f}%"


def _kelly(fraction: float | None) -> str:
    """§M24 honest rendering of the (already-capped) Kelly fraction.

    None  → no usable odds (couldn't price);
    0.0   → priced but edge below the bet threshold (no bet);
    >0    → the capped stake as a percent of bankroll.
    """
    if fraction is None:
        return f"{_NO_MARKET} (couldn't price)"
    if fraction <= 0.0:
        return "no bet (edge below threshold)"
    return f"{fraction * 100:.2f}% of bankroll"
