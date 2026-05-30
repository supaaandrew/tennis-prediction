"""Cross-source utility modules.

`score_parser` is the canonical score-string parser shared by Sackmann and
matchstat (§T4) — neither source owns it because both serialize scores in
the same wire format.
"""

from __future__ import annotations

from tennis.utils.score_parser import ParsedScore, ScoreParseError, parse_score

__all__ = ["ParsedScore", "ScoreParseError", "parse_score"]
