"""Pydantic v2 response DTOs for the matchstat API.

One DTO per response shape we consume. Field names match the API verbatim
(`countryAcr`, `roundId`, `seed1`, `odd1`, `tournamentId`, `hasNextPage`).
`extra='ignore'` lets the upstream evolve without breaking us.

§T7 string-typed fields land their parsing here so downstream callers never
see the raw upstream type — they get `Decimal | None` for odds, an `int |
None` parsed `seed_numeric` next to the verbatim `seed1` string, etc.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SEED_CODES: frozenset[str] = frozenset(("WC", "Q", "LL", "PR", "ALT"))


def _opt_decimal(v: Any) -> Decimal | None:
    """§T7 — odds arrive as strings (`"2.10"`, `""`, or null). Empty / null → None."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    if isinstance(v, str):
        stripped = v.strip()
        if not stripped:
            return None
        try:
            return Decimal(stripped)
        except InvalidOperation:
            return None
    return None


def _opt_int(v: Any) -> int | None:
    """`best_of` arrives as `"3"` / `"5"` / null / empty. Coerce safely."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        stripped = v.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _opt_date(v: Any) -> date | None:
    """`birthday` arrives as `"YYYY-MM-DD"` / null / empty. Coerce safely."""
    if v is None:
        return None
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        stripped = v.strip()
        if not stripped:
            return None
        try:
            return date.fromisoformat(stripped[:10])
        except ValueError:
            return None
    return None


class _MsBase(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


# ---------------------------------------------------------------------------
# Lookup tables (§T5 / §T6 lookup-cache)
# ---------------------------------------------------------------------------
class MsCourt(_MsBase):
    id: int
    name: str | None = None


class MsRound(_MsBase):
    id: int
    name: str | None = None


class MsRank(_MsBase):
    id: int
    name: str | None = None


class MsCountry(_MsBase):
    id: int | None = None
    acr: str | None = None
    name: str | None = None


# ---------------------------------------------------------------------------
# Tournament shapes (calendar + info + results)
# ---------------------------------------------------------------------------
class MsTournament(_MsBase):
    # Identity
    id: int | None = Field(default=None, alias="tournamentId")
    seasonId: int | None = None
    name: str | None = None
    site: str | None = None
    countryAcr: str | None = None
    # §T12 venue coords (calendar enrichment).
    latitude: float | None = None
    longitude: float | None = None
    # Surface / tier joined via include=tournament.court / include=tournament.rank
    court: MsCourt | None = None
    rank: MsRank | None = None
    # Schedule
    dateStart: str | None = None  # "YYYY-MM-DD"
    dateEnd: str | None = None
    drawSize: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_id_alias(cls, data: Any) -> Any:
        """Accept both `id` and `tournamentId` — calendar uses one, fixture-include the other."""
        if isinstance(data, dict) and "tournamentId" not in data and "id" in data:
            data = {**data, "tournamentId": data["id"]}
        return data


class MsTournamentInfo(_MsBase):
    tournament: MsTournament
    season: int | None = None


class MsTournamentResultMatch(_MsBase):
    """A historical (past) match row from /tournament/results or /past-matches.
    **§T3 convention: `player1` here is ALWAYS the winner.**"""

    id: int | None = None
    tournamentId: int | None = None
    roundId: int | None = None
    player1Id: int | None = None
    player2Id: int | None = None
    player1: dict[str, Any] | None = None
    player2: dict[str, Any] | None = None
    result: str | None = None  # full score string
    bestOf: int | None = None


# ---------------------------------------------------------------------------
# Player shapes
# ---------------------------------------------------------------------------
class MsPlayer(_MsBase):
    id: int | None = None
    name: str | None = None
    countryAcr: str | None = None
    birthday: date | None = None

    _val_birthday = field_validator("birthday", mode="before")(
        classmethod(lambda cls, v: _opt_date(v))
    )


class MsPlayerProfile(_MsBase):
    id: int
    name: str | None = None
    countryAcr: str | None = None
    birthday: date | None = None
    rightHanded: bool | None = None
    twoHandedBackhand: bool | None = None
    currentRank: int | None = None
    currentPoints: int | None = None

    _val_birthday = field_validator("birthday", mode="before")(
        classmethod(lambda cls, v: _opt_date(v))
    )


# ---------------------------------------------------------------------------
# Fixture (§T3 convention: `player1` here is FIRST-LISTED, NOT winner —
# fixtures are forward-looking; result == "" until played).
# ---------------------------------------------------------------------------
class MsFixture(_MsBase):
    id: int = Field(alias="id")  # the fixture / match id within matchstat
    tournamentId: int | None = None
    roundId: int | None = None
    tournament: MsTournament | None = None
    round: MsRound | None = None
    # Players
    player1Id: int | None = None
    player2Id: int | None = None
    player1: MsPlayer | None = None
    player2: MsPlayer | None = None
    # §T7 — strings on the wire
    odd1: Decimal | None = None
    odd2: Decimal | None = None
    seed1: str | None = None
    seed2: str | None = None
    seed_numeric_1: int | None = None
    seed_numeric_2: int | None = None
    best_of: int | None = Field(default=None, alias="bestOf")
    # Schedule / state
    dateStart: str | None = None
    timeStart: str | None = None
    live: bool | None = None
    result: str | None = None  # "" on fixtures; populated on past-matches
    # h2h.player{1,2}Wins lands here when fetched via include=h2h.
    h2h: dict[str, Any] | None = None

    _val_odd1 = field_validator("odd1", mode="before")(
        classmethod(lambda cls, v: _opt_decimal(v))
    )
    _val_odd2 = field_validator("odd2", mode="before")(
        classmethod(lambda cls, v: _opt_decimal(v))
    )
    _val_best_of = field_validator("best_of", mode="before")(
        classmethod(lambda cls, v: _opt_int(v))
    )

    @model_validator(mode="after")
    def _derive_seed_numeric(self) -> MsFixture:
        # §T7 — keep seed verbatim AND expose a parsed int for the cases
        # numeric. Codes "WC|Q|LL|PR|ALT" → None, never `0`.
        object.__setattr__(self, "seed_numeric_1", _parse_seed(self.seed1))
        object.__setattr__(self, "seed_numeric_2", _parse_seed(self.seed2))
        return self


def _parse_seed(raw: str | None) -> int | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.upper() in _SEED_CODES:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Rankings (§T13 — date-filtered Top-N seed)
# ---------------------------------------------------------------------------
class MsRankingRow(_MsBase):
    playerId: int
    player: MsPlayer | None = None
    currentRank: int
    points: int | None = None
    rankingDate: str | None = None  # "YYYY-MM-DD"


# ---------------------------------------------------------------------------
# Search (§T11 — DOB tiebreaker enrichment)
# ---------------------------------------------------------------------------
class MsSearchResult(_MsBase):
    id: int | None = None
    name: str | None = None
    countryAcr: str | None = None
    birthday: date | None = None
    type: str | None = None  # "player" / "tournament" — caller filters

    _val_birthday = field_validator("birthday", mode="before")(
        classmethod(lambda cls, v: _opt_date(v))
    )


# ---------------------------------------------------------------------------
# H2H stats (§T14 — clutch fields)
# ---------------------------------------------------------------------------
class _MsH2hSideStats(_MsBase):
    decidingSetWin: int | None = None
    tiebreakWon: int | None = None
    firstSetLoseMatchWin: int | None = None
    matchesWon: int | None = None


class MsH2hStats(_MsBase):
    matchesCount: int = 0
    player1Stats: _MsH2hSideStats | None = None
    player2Stats: _MsH2hSideStats | None = None


# ---------------------------------------------------------------------------
# Paginated envelope shape (every list endpoint returns this shape)
# ---------------------------------------------------------------------------
class MsPage(_MsBase):
    """Generic page envelope. The data payload is `data: list[...]`; consumers
    cast / `model_validate` each element into the appropriate DTO."""

    data: list[Any] = Field(default_factory=list)
    hasNextPage: bool = False
    pageNo: int | None = None
    pageSize: int | None = None
