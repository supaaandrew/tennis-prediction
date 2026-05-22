"""Deterministic ID hashing — single source of truth for entity identity.

All IDs are SHA-256-derived positive 63-bit integers, safe to store in a
Postgres BIGINT (which is signed 64-bit). The same logical entity, observed
through any source, hashes to the same ID. This is what makes cross-source
deduplication work without a reconciliation table.

p1/p2 perspective rule
----------------------
Per locked decision: `match_id % 2` picks which of the two players is treated
as "p1" in the feature row. The rule is applied to a *sorted* pair of player
IDs so the assignment is fully deterministic and reproducible — there is
never any randomness at runtime.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, date, datetime
from typing import Any

_BIT_MASK = (1 << 63) - 1
_SEPARATOR = "\x1f"  # ASCII unit separator — won't appear in normalized fields


def _canonicalize(value: Any) -> str:
    """Convert a value to a canonical string for hashing.

    Rules:
      - None → ``"\x00"``
      - datetime: must be tz-aware; converted to UTC ISO-8601
      - date: ISO-8601
      - str: trimmed + lower-cased
      - everything else: str()
    """
    if value is None:
        return "\x00"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware for stable hashing")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool):
        # bool is a subclass of int — handle FIRST and canonicalize distinctly
        # so a tuple containing True never collides with one containing 1.
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return str(value)


def stable_hash_int63(parts: tuple[Any, ...]) -> int:
    """SHA-256 of canonicalized parts, truncated to a positive 63-bit int.

    Two callers passing the same logical tuple always get the same id, across
    machines, processes, and Python versions.
    """
    canonical = _SEPARATOR.join(_canonicalize(p) for p in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _BIT_MASK


# ---------------------------------------------------------------------------
# Entity ID helpers
# ---------------------------------------------------------------------------
def player_id_from_source(*, source: str, source_uid: str) -> int:
    """Stable player ID derived from a source-scoped UID.

    Shadow players (created by the ATP scraper before Sackmann publishes them)
    get a stable hashed ID this way and KEEP it forever — when Sackmann
    eventually catches up, the Sackmann atp_id is stored as an alias rather
    than replacing the canonical player_id.
    """
    return stable_hash_int63(("player", source, source_uid))


def venue_id(*, city: str, country_code: str) -> int:
    return stable_hash_int63(("venue", city, country_code))


def tournament_id(*, season: int, slug: str) -> int:
    return stable_hash_int63(("tournament", season, slug))


def match_id(
    *,
    tournament_id: int,
    round: str,
    player_a: int,
    player_b: int,
    match_date: date,
) -> int:
    """Deterministic match ID.

    Identity is `(tournament_id, round, player_a, player_b, match_date)` with
    player order canonicalized (sorted) before hashing. `start_ts` is
    DELIBERATELY NOT part of the hash: it is mutable metadata on the row
    (NULL for historical Sackmann rows, a real timestamp once known). If we
    included it, the same logical match would hash to two different IDs
    depending on whether the historical or live ingest landed first — exactly
    the cross-source dedup bug stable hashes are supposed to prevent.

    Edge case: a match retired-and-replayed on the same date in the same
    round between the same two players collides. This is so rare in singles
    play that we accept it; if it ever happens, the application logs a
    `dead_letter` row when the upsert sees a stats conflict.
    """
    a, b = sorted((player_a, player_b))
    return stable_hash_int63(("match", tournament_id, round, a, b, match_date))


# ---------------------------------------------------------------------------
# p1/p2 perspective — the locked decision
# ---------------------------------------------------------------------------
def p1_player_id(match_id_value: int, player_a: int, player_b: int) -> int:
    """Return the player ID that occupies the "p1" slot for this match.

    Rule (locked):
      - Sort the two player IDs.
      - If match_id is even, p1 = smaller ID; if odd, p1 = larger ID.

    Why: deterministic, balanced ~50/50 because match_id is a uniformly
    distributed hash, and reproducible — no runtime randomness, no
    dependency on which source listed the match first.
    """
    smaller, larger = sorted((player_a, player_b))
    return smaller if (match_id_value % 2) == 0 else larger


def p2_player_id(match_id_value: int, player_a: int, player_b: int) -> int:
    """Counterpart to `p1_player_id`."""
    smaller, larger = sorted((player_a, player_b))
    return larger if (match_id_value % 2) == 0 else smaller


def is_p1(match_id_value: int, player_id: int, opponent_id: int) -> bool:
    """Predicate form: True iff `player_id` is the p1 for this match."""
    return p1_player_id(match_id_value, player_id, opponent_id) == player_id


# ---------------------------------------------------------------------------
# Player name normalization (cross-source identity resolution)
# ---------------------------------------------------------------------------
#
# Latin-extended letters that do NOT decompose under NFKD. We map them
# explicitly so e.g. "Đoković" folds to "djokovic" before fuzzy matching.
# Add to this map only on confirmed real-world player names — don't speculate.
_LATIN_FOLDS: dict[str, str] = {
    # Serbian "Đ/đ" romanizes as "Dj/dj" in English-language tennis coverage
    # (e.g. "Đoković" -> "Djokovic"). The example in the locked spec pins this.
    "Đ": "Dj", "đ": "dj",
    "Ð": "D",  "ð": "d",   # Icelandic eth (distinct from Serbian Đ)
    "Ł": "L",  "ł": "l",   # Polish L with stroke
    "Ø": "O",  "ø": "o",   # Nordic O with stroke
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "Þ": "TH", "þ": "th",
    "ß": "ss",
}

_PUNCT_TO_SPACE = re.compile(r"[^a-z0-9\s]")
_COLLAPSE_WS = re.compile(r"\s+")


def _apply_latin_folds(s: str) -> str:
    return "".join(_LATIN_FOLDS.get(c, c) for c in s)


def normalize_player_name(raw: str) -> str:
    """Normalize a player name to a canonical alias key.

    Rules (applied in order):
      1. Apply Latin-extended folds (Đ→D, ø→o, etc.) — covers letters that
         NFKD does NOT decompose.
      2. NFKD-decompose the rest; strip combining marks (ö→o, ć→c, é→e).
      3. Lowercase.
      4. Replace any non-`[a-z0-9\\s]` character with a space (strips
         periods in initials, apostrophes, hyphens, etc.).
      5. Collapse runs of whitespace; strip leading/trailing whitespace.

    Examples:
      "Novak Djokovic"  → "novak djokovic"
      "N. Djokovic"     → "n djokovic"
      "Đoković N."      → "djokovic n"
      "NDjokovic"       → "ndjokovic"
    """
    if not isinstance(raw, str):
        raise TypeError(f"normalize_player_name requires str, got {type(raw).__name__}")
    s = _apply_latin_folds(raw)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_TO_SPACE.sub(" ", s)
    s = _COLLAPSE_WS.sub(" ", s).strip()
    return s
