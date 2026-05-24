"""Player identity resolution (the locked 4-tier chain).

Implements DECISIONS.md §11 / §15.6:

  1. Exact match on normalized alias in `player_aliases`.
  2. DOB + country_code tiebreaker against known players.
  3. Fuzzy match (rapidfuzz `token_set_ratio`) >= `fuzzy_threshold`;
     requires a DOB match when `require_dob_for_fuzzy` is True.
  4. Manual override from `config/player_overrides.yaml`.

If all four tiers fail and the reference carries a Sackmann `atp_id`, a
shadow player is created (its hashed `player_id` is kept forever, A9).
A name-only reference with no `atp_id` raises `PlayerResolutionError`
so the adapter can route the row to `dead_letter`.

`players.aliases` JSONB is deprecated (H6) — only the `player_aliases`
table is written here.

The DOB/fuzzy tiers operate over an in-memory candidate index built by
`register()` (the adapter registers every `atp_players.csv` row before
processing matches). This keeps the resolver decoupled from any
DOB/fuzzy query method on the repository Protocols.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

from tennis.core.errors import PlayerResolutionError
from tennis.core.ids import normalize_player_name, player_id_from_source
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import (
    PlayerAliasRepository,
    PlayerRepository,
)
from tennis.storage.postgres.rows import Confidence, DominantHand, PlayerAliasRow, PlayerRow

_SACKMANN = "sackmann"


@dataclass(frozen=True, slots=True)
class PlayerRef:
    """A player reference to resolve to a canonical `player_id`."""

    name: str
    source: str
    source_uid: str | None = None  # atp_id, when the source provides one
    dob: date | None = None
    country_code: str | None = None
    dominant_hand: DominantHand | None = None
    height_cm: int | None = None


@dataclass(frozen=True, slots=True)
class OverrideEntry:
    """One hand-curated alias mapping from `player_overrides.yaml`."""

    alias: str  # normalized
    source: str
    sackmann_atp_id: str
    dob: date | None = None
    country_code: str | None = None


def load_overrides(path: str | Path) -> tuple[OverrideEntry, ...]:
    """Load and normalize `player_overrides.yaml`. Empty file -> ()."""
    p = Path(path)
    if not p.is_file():
        return ()
    with p.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    entries: list[OverrideEntry] = []
    for item in raw.get("overrides", []) or []:
        atp_id = item.get("sackmann_atp_id")
        alias = item.get("alias")
        source = item.get("source")
        if not atp_id or not alias or not source:
            continue
        dob_raw = item.get("dob")
        entries.append(
            OverrideEntry(
                alias=normalize_player_name(str(alias)),
                source=str(source),
                sackmann_atp_id=str(atp_id),
                dob=date.fromisoformat(str(dob_raw)) if dob_raw else None,
                country_code=item.get("country_code"),
            )
        )
    return tuple(entries)


class SackmannResolver:
    """Resolves player references to canonical `player_id`s.

    Construct once per ingest run; call `register()` for every known
    player (from `atp_players.csv`) before resolving match references.
    """

    def __init__(
        self,
        *,
        players: PlayerRepository,
        aliases: PlayerAliasRepository,
        fuzzy_threshold: float = 0.92,
        require_dob_for_fuzzy: bool = True,
        overrides: Iterable[OverrideEntry] = (),
    ) -> None:
        self._players = players
        self._aliases = aliases
        self._cutoff = fuzzy_threshold * 100.0
        self._require_dob_for_fuzzy = require_dob_for_fuzzy
        self._overrides: dict[tuple[str, str], OverrideEntry] = {
            (o.alias, o.source): o for o in overrides
        }
        # In-memory candidate indexes for tiers 2 and 3.
        self._dob_country_index: dict[tuple[date, str], list[int]] = defaultdict(list)
        self._fuzzy_entries: list[tuple[str, int, date | None]] = []
        # Player IDs already registered — guards against duplicate index
        # entries (which would otherwise make tier-2 see false "ambiguous").
        self._registered_ids: set[int] = set()
        self._logger = get_logger("tennis.adapters.sackmann.resolver")

    # -- registration -------------------------------------------------------
    def register(self, player: PlayerRow) -> None:
        """Index a known player and write its exact alias.

        Idempotent per player: re-registering the same `player_id` (e.g. a
        re-ingest) is a no-op, so the candidate indexes never accumulate
        duplicates.

        Collision-safe: if a *different* player already owns the normalized
        full-name alias (two ATP IDs sharing one normalized name, e.g.
        "J. Johansson"), the existing mapping is NOT overwritten. The new
        player instead gets an alias keyed on its unique `source_uid` and a
        warning is logged; the ambiguous name then falls through to the
        DOB+country tier for the second player.
        """
        if player.player_id in self._registered_ids:
            return  # already registered — fully idempotent
        self._registered_ids.add(player.player_id)

        norm = normalize_player_name(player.full_name)
        self._index(norm, player.player_id, player.date_of_birth, player.country_code)

        existing = self._aliases.get(alias=norm, source=_SACKMANN)
        if existing is not None and existing.player_id != player.player_id:
            self._logger.warning(
                "sackmann_alias_collision",
                alias=norm,
                existing_player_id=existing.player_id,
                new_player_id=player.player_id,
                source_uid=player.source_uid,
            )
            # source_uid is unique per player, so this alias never collides.
            self._upsert_alias(
                alias=player.source_uid,
                source=_SACKMANN,
                player_id=player.player_id,
                confidence="exact",
                dob=player.date_of_birth,
                country_code=player.country_code,
            )
            return

        self._upsert_alias(
            alias=norm,
            source=_SACKMANN,
            player_id=player.player_id,
            confidence="exact",
            dob=player.date_of_birth,
            country_code=player.country_code,
        )

    # -- resolution ---------------------------------------------------------
    def resolve(self, ref: PlayerRef) -> int:
        """Resolve `ref` to a canonical `player_id` via the 4-tier chain.

        Raises `PlayerResolutionError` if every tier fails and no shadow
        player can be created (name-only reference without an atp_id).
        """
        norm = normalize_player_name(ref.name)

        # Tier 1 — exact alias.
        existing = self._aliases.get(alias=norm, source=ref.source)
        if existing is not None:
            return existing.player_id

        # Tier 2 — DOB + country tiebreaker.
        if ref.dob is not None and ref.country_code:
            candidates = self._dob_country_index.get((ref.dob, ref.country_code), [])
            if len(candidates) == 1:
                pid = candidates[0]
                self._upsert_alias(
                    alias=norm,
                    source=ref.source,
                    player_id=pid,
                    confidence="exact",
                    dob=ref.dob,
                    country_code=ref.country_code,
                )
                return pid

        # Tier 3 — fuzzy.
        pid = self._fuzzy_resolve(norm, ref)
        if pid is not None:
            self._upsert_alias(
                alias=norm,
                source=ref.source,
                player_id=pid,
                confidence="fuzzy",
                dob=ref.dob,
                country_code=ref.country_code,
            )
            return pid

        # Tier 4 — manual override.
        entry = self._overrides.get((norm, ref.source))
        if entry is not None:
            pid = player_id_from_source(source=_SACKMANN, source_uid=entry.sackmann_atp_id)
            self._upsert_alias(
                alias=norm,
                source=ref.source,
                player_id=pid,
                confidence="manual",
                dob=ref.dob or entry.dob,
                country_code=ref.country_code or entry.country_code,
            )
            return pid

        # Shadow player — only with a Sackmann atp_id (A9).
        if ref.source == _SACKMANN and ref.source_uid:
            return self._create_shadow(norm, ref)

        raise PlayerResolutionError(
            f"could not resolve player name={ref.name!r} source={ref.source!r} "
            f"(no alias / DOB+country / fuzzy / override match, no atp_id for shadow)"
        )

    # -- internals ----------------------------------------------------------
    def _fuzzy_resolve(self, norm: str, ref: PlayerRef) -> int | None:
        if self._require_dob_for_fuzzy:
            if ref.dob is None:
                return None
            entries = [(n, pid) for (n, pid, dob) in self._fuzzy_entries if dob == ref.dob]
        else:
            entries = [(n, pid) for (n, pid, _dob) in self._fuzzy_entries]
        if not entries:
            return None
        choices = [n for (n, _pid) in entries]
        result = process.extractOne(
            norm, choices, scorer=fuzz.token_set_ratio, score_cutoff=self._cutoff
        )
        if result is None:
            return None
        _choice, _score, idx = result
        return entries[idx][1]

    def _create_shadow(self, norm: str, ref: PlayerRef) -> int:
        assert ref.source_uid is not None  # guarded by caller
        pid = player_id_from_source(source=_SACKMANN, source_uid=ref.source_uid)
        self._players.upsert(
            PlayerRow(
                player_id=pid,
                full_name=ref.name,
                source=_SACKMANN,
                source_uid=ref.source_uid,
                country_code=ref.country_code,
                date_of_birth=ref.dob,
                dominant_hand=ref.dominant_hand,
                height_cm=ref.height_cm,
            )
        )
        self._index(norm, pid, ref.dob, ref.country_code)
        self._upsert_alias(
            alias=norm,
            source=_SACKMANN,
            player_id=pid,
            confidence="exact",
            dob=ref.dob,
            country_code=ref.country_code,
        )
        return pid

    def _index(
        self, norm: str, player_id: int, dob: date | None, country_code: str | None
    ) -> None:
        if dob is not None and country_code:
            self._dob_country_index[(dob, country_code)].append(player_id)
        self._fuzzy_entries.append((norm, player_id, dob))

    def _upsert_alias(
        self,
        *,
        alias: str,
        source: str,
        player_id: int,
        confidence: Confidence,
        dob: date | None,
        country_code: str | None,
    ) -> None:
        self._aliases.upsert(
            PlayerAliasRow(
                alias=alias,
                source=source,
                player_id=player_id,
                confidence=confidence,
                dob=dob,
                country_code=country_code,
            )
        )
