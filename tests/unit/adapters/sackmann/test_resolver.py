"""Unit tests for the 4-tier player resolver (mocked repositories)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from tennis.adapters.sackmann.resolver import (
    OverrideEntry,
    PlayerRef,
    SackmannResolver,
)
from tennis.core.errors import PlayerResolutionError
from tennis.core.ids import player_id_from_source
from tennis.storage.postgres.rows import PlayerAliasRow, PlayerRow


# ---------------------------------------------------------------------------
# In-memory fakes for the two repositories the resolver touches
# ---------------------------------------------------------------------------
class FakeAliasRepo:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], PlayerAliasRow] = {}

    def get(self, *, alias: str, source: str) -> PlayerAliasRow | None:
        return self.store.get((alias, source))

    def list_for_player(self, player_id: int):  # pragma: no cover - unused
        return [r for r in self.store.values() if r.player_id == player_id]

    def upsert(self, row: PlayerAliasRow) -> PlayerAliasRow:
        self.store[(row.alias, row.source)] = row
        return row


class FakePlayerRepo:
    def __init__(self) -> None:
        self.by_id: dict[int, PlayerRow] = {}
        self.upserts: list[PlayerRow] = []

    def get(self, player_id: int) -> PlayerRow | None:
        return self.by_id.get(player_id)

    def get_by_source(self, *, source: str, source_uid: str) -> PlayerRow | None:
        for row in self.by_id.values():
            if row.source == source and row.source_uid == source_uid:
                return row
        return None

    def upsert(self, row: PlayerRow) -> PlayerRow:
        self.by_id[row.player_id] = row
        self.upserts.append(row)
        return row


def _player(player_id: int, name: str, uid: str, **kw: object) -> PlayerRow:
    return PlayerRow(
        player_id=player_id, full_name=name, source="sackmann", source_uid=uid, **kw
    )


def _resolver(players: FakePlayerRepo, aliases: FakeAliasRepo, **kw: object) -> SackmannResolver:
    defaults: dict[str, object] = {"fuzzy_threshold": 0.92, "require_dob_for_fuzzy": True}
    defaults.update(kw)
    return SackmannResolver(players=players, aliases=aliases, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tier 1 — exact alias
# ---------------------------------------------------------------------------
def test_tier1_exact_alias_no_fuzzy_call() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    aliases.store[("novak djokovic", "sackmann")] = PlayerAliasRow(
        alias="novak djokovic", source="sackmann", player_id=999, confidence="exact"
    )
    r = _resolver(players, aliases)
    r._fuzzy_resolve = MagicMock(side_effect=AssertionError("fuzzy must not run"))  # type: ignore[method-assign]

    pid = r.resolve(PlayerRef(name="Novak Djokovic", source="sackmann", source_uid="104925"))

    assert pid == 999
    r._fuzzy_resolve.assert_not_called()


# ---------------------------------------------------------------------------
# Tier 2 — DOB + country tiebreaker
# ---------------------------------------------------------------------------
def test_tier2_dob_country_match() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases)
    r.register(_player(222, "Rafael Nadal", "2", date_of_birth=date(1986, 6, 3), country_code="ESP"))

    # Different name (tier-1 miss) but matching DOB + country.
    pid = r.resolve(
        PlayerRef(name="R. Nadal", source="sackmann", dob=date(1986, 6, 3), country_code="ESP")
    )

    assert pid == 222
    assert aliases.store[("r nadal", "sackmann")].confidence == "exact"


def test_tier2_ambiguous_dob_country_falls_through() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases, require_dob_for_fuzzy=True)
    dob, ioc = date(1990, 1, 1), "USA"
    r.register(_player(1, "Player One", "1", date_of_birth=dob, country_code=ioc))
    r.register(_player(2, "Player Two", "2", date_of_birth=dob, country_code=ioc))

    # Two candidates share DOB+country -> tiebreaker can't resolve -> shadow.
    pid = r.resolve(PlayerRef(name="Unknown Name", source="sackmann", source_uid="77", dob=dob, country_code=ioc))
    assert pid == player_id_from_source(source="sackmann", source_uid="77")


# ---------------------------------------------------------------------------
# Tier 3 — fuzzy
# ---------------------------------------------------------------------------
def test_tier3_fuzzy_above_threshold_writes_fuzzy_alias() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases, require_dob_for_fuzzy=False)
    r.register(_player(333, "Novak Djokovic", "3"))

    # Token reorder scores 100 on token_set_ratio (tier-1 misses the alias).
    pid = r.resolve(PlayerRef(name="Djokovic Novak", source="sackmann", source_uid="3"))

    assert pid == 333
    assert aliases.store[("djokovic novak", "sackmann")].confidence == "fuzzy"


def test_tier3_fuzzy_below_threshold_falls_through_and_raises() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases, require_dob_for_fuzzy=False)
    r.register(_player(444, "Roger Federer", "4"))

    # No alias, no DOB, fuzzy too low, no override, no atp_id -> raise.
    with pytest.raises(PlayerResolutionError):
        r.resolve(PlayerRef(name="Marin Cilic", source="sackmann", source_uid=None))


def test_tier3_require_dob_skips_fuzzy_when_ref_has_no_dob() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases, require_dob_for_fuzzy=True)
    r.register(_player(555, "Novak Djokovic", "5"))

    ref = PlayerRef(name="Djokovic Novak", source="sackmann", source_uid="5", dob=None)
    # Direct: fuzzy yields nothing because DOB is required and absent.
    assert r._fuzzy_resolve("djokovic novak", ref) is None

    # End-to-end: resolution falls to shadow (NOT the fuzzy candidate 555).
    pid = r.resolve(ref)
    assert pid == player_id_from_source(source="sackmann", source_uid="5")
    assert pid != 555


# ---------------------------------------------------------------------------
# Tier 4 — manual override
# ---------------------------------------------------------------------------
def test_tier4_manual_override() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    overrides = [OverrideEntry(alias="ndjokovic", source="sackmann", sackmann_atp_id="104925")]
    r = _resolver(players, aliases, require_dob_for_fuzzy=True, overrides=overrides)

    pid = r.resolve(PlayerRef(name="NDjokovic", source="sackmann", source_uid=None))

    assert pid == player_id_from_source(source="sackmann", source_uid="104925")
    assert aliases.store[("ndjokovic", "sackmann")].confidence == "manual"


# ---------------------------------------------------------------------------
# Failure + shadow creation
# ---------------------------------------------------------------------------
def test_all_tiers_fail_raises_when_no_atp_id() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases)
    with pytest.raises(PlayerResolutionError):
        r.resolve(PlayerRef(name="Nobody Here", source="sackmann", source_uid=None))


def test_shadow_player_created_from_atp_id() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases)

    pid = r.resolve(PlayerRef(name="Brand New Player", source="sackmann", source_uid="999999"))

    expected = player_id_from_source(source="sackmann", source_uid="999999")
    assert pid == expected
    assert players.upserts[-1].player_id == expected
    assert players.upserts[-1].full_name == "Brand New Player"
    assert ("brand new player", "sackmann") in aliases.store


# ---------------------------------------------------------------------------
# register() idempotency + alias-collision safety (FIX 2)
# ---------------------------------------------------------------------------
def test_register_twice_same_player_is_idempotent() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases)
    p = _player(111, "Rafael Nadal", "2", date_of_birth=date(1986, 6, 3), country_code="ESP")

    r.register(p)
    r.register(p)  # re-ingest — must not duplicate the alias or the index

    # Exactly one alias row for the normalized name.
    assert len(aliases.store) == 1
    assert aliases.store[("rafael nadal", "sackmann")].player_id == 111

    # The DOB+country index is not duplicated, so tier-2 still resolves
    # unambiguously (a doubled index would look like two candidates).
    pid = r.resolve(
        PlayerRef(name="R. Nadal", source="sackmann", dob=date(1986, 6, 3), country_code="ESP")
    )
    assert pid == 111


def test_register_collision_does_not_overwrite_alias() -> None:
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases)
    r._logger = MagicMock()  # type: ignore[assignment]

    # Two distinct ATP IDs whose names normalize identically.
    r.register(_player(111, "Joachim Johansson", "1"))
    r.register(_player(222, "Joachim Johansson", "2"))

    # The normalized-name alias still maps to the FIRST player.
    assert aliases.store[("joachim johansson", "sackmann")].player_id == 111
    # The second player gets a unique source_uid-keyed alias instead.
    assert aliases.store[("2", "sackmann")].player_id == 222
    # And the collision was logged.
    r._logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# §T11 — matchstat search enrichment (DOB+country tier extension)
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402 — local-only, after the existing tests


class _FakeSearchEnricher:
    """Records enrich() calls and returns a programmable hit (or None)."""

    def __init__(self, hit: SimpleNamespace | None) -> None:
        self._hit = hit
        self.calls: list[str] = []

    def enrich(self, *, name: str) -> SimpleNamespace | None:
        self.calls.append(name)
        return self._hit


def _ms_hit(*, birthday: date | None, countryAcr: str | None) -> SimpleNamespace:
    """A pydantic-shaped stand-in for `MsSearchResult` (only the two fields
    the resolver consumes)."""
    return SimpleNamespace(birthday=birthday, countryAcr=countryAcr)


def test_t11_search_enrichment_locks_tier2_on_unique_dob_country_match() -> None:
    """A matchstat hit whose DOB+country resolves to exactly ONE indexed
    player locks the resolution at §C2 Tier 2 with `confidence='exact'`,
    even though the inbound `ref.dob` is None (Sackmann gap)."""
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    enricher = _FakeSearchEnricher(
        hit=_ms_hit(birthday=date(1986, 6, 3), countryAcr="ESP")
    )
    r = _resolver(players, aliases, search_enricher=enricher)
    r.register(_player(222, "Rafael Nadal", "2",
                       date_of_birth=date(1986, 6, 3), country_code="ESP"))

    # Reference name doesn't tier-1, has no DOB (would short-circuit Tier 3).
    pid = r.resolve(PlayerRef(name="R. Nadal", source="sackmann"))

    assert pid == 222
    assert enricher.calls == ["R. Nadal"]
    alias = aliases.store[("r nadal", "sackmann")]
    assert alias.confidence == "exact"
    assert alias.dob == date(1986, 6, 3)
    assert alias.country_code == "ESP"


def test_t11_search_enrichment_dob_divergence_falls_through() -> None:
    """A matchstat hit whose DOB+country does NOT match any registered
    player (0 candidates) falls through without writing the enriched
    alias — divergence signals the wrong player despite the name match.
    The resolver then runs its usual remaining tiers; with no fuzzy match
    and no atp_id the canonical failure (PlayerResolutionError) fires."""
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    enricher = _FakeSearchEnricher(
        # Matchstat returns Federer-like DOB+country, divergent from Nadal.
        hit=_ms_hit(birthday=date(1981, 8, 8), countryAcr="SUI")
    )
    r = _resolver(players, aliases, search_enricher=enricher)
    r.register(_player(222, "Rafael Nadal", "2",
                       date_of_birth=date(1986, 6, 3), country_code="ESP"))

    with pytest.raises(PlayerResolutionError):
        r.resolve(PlayerRef(name="R. Nadal", source="sackmann"))

    # The enricher was consulted exactly once; its hit did NOT lock an alias
    # under "exact" (no `("r nadal", "sackmann")` entry from the §T11 branch).
    assert enricher.calls == ["R. Nadal"]
    assert ("r nadal", "sackmann") not in aliases.store


def test_t11_search_enricher_none_is_clean_noop() -> None:
    """Training without `MATCHSTAT_API_KEY` -> `search_enricher=None`:
    behavior is bit-for-bit today's Sackmann-only path. With no DOB on
    the ref and no atp_id, the existing failure mode (PlayerResolutionError)
    fires exactly as before the §T11 hook landed."""
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    r = _resolver(players, aliases, search_enricher=None)
    r.register(_player(333, "Novak Djokovic", "3"))

    with pytest.raises(PlayerResolutionError):
        r.resolve(PlayerRef(name="N. Djokovic", source="sackmann"))
    # And no alias was ever written for the unresolved reference name.
    assert ("n djokovic", "sackmann") not in aliases.store


def test_t11_enrich_returns_none_falls_through() -> None:
    """A None enrich() return (404 / quota / parse error — absorbed in
    the sidecar) must NOT raise; resolver continues to fuzzy / shadow."""
    players, aliases = FakePlayerRepo(), FakeAliasRepo()
    enricher = _FakeSearchEnricher(hit=None)
    r = _resolver(players, aliases, search_enricher=enricher)
    # No registered player, no atp_id, no DOB → existing failure path raises.
    with pytest.raises(PlayerResolutionError):
        r.resolve(PlayerRef(name="Unknown Player", source="sackmann"))
    # The enricher was consulted (one call), and the resolver did NOT swallow
    # the eventual PlayerResolutionError.
    assert enricher.calls == ["Unknown Player"]
