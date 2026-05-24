"""Sackmann GitHub mirror adapter (primary historical source).

Reads the locally-mirrored `JeffSackmann/tennis_atp` CSVs, parses raw
fields into Row DTOs (`parser`), resolves player identities
(`resolver`), and writes matches / stats / rankings / players through
the storage repositories (`adapter`).

See DECISIONS.md §15.1 (field → column mapping) and §15.6 (cross-source
reconciliation) for the contract this adapter implements.
"""

from tennis.adapters.sackmann.adapter import (
    FilesystemMirrorReader,
    MirrorReader,
    SackmannAdapter,
)
from tennis.adapters.sackmann.parser import (
    ParsedMatch,
    ParsedPlayer,
    ParsedPlayerRef,
    ParsedRanking,
    ParsedStats,
    ParsedTournament,
    ScoreInfo,
    map_tier,
    parse_best_of,
    parse_date,
    parse_hand,
    parse_height,
    parse_int,
    parse_match,
    parse_minutes,
    parse_player,
    parse_ranking,
    parse_score,
    parse_surface,
    slugify,
)
from tennis.adapters.sackmann.resolver import (
    OverrideEntry,
    PlayerRef,
    SackmannResolver,
    load_overrides,
)

__all__ = [
    # parser
    "ParsedMatch",
    "ParsedPlayer",
    "ParsedPlayerRef",
    "ParsedRanking",
    "ParsedStats",
    "ParsedTournament",
    "ScoreInfo",
    "map_tier",
    "parse_best_of",
    "parse_date",
    "parse_hand",
    "parse_height",
    "parse_int",
    "parse_match",
    "parse_minutes",
    "parse_player",
    "parse_ranking",
    "parse_score",
    "parse_surface",
    "slugify",
    # resolver
    "OverrideEntry",
    "PlayerRef",
    "SackmannResolver",
    "load_overrides",
    # adapter
    "SackmannAdapter",
    "MirrorReader",
    "FilesystemMirrorReader",
]
