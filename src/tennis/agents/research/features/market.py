"""Market-signal feature family (R7) — `features.market` (§15.5 + §M19).

`MarketExtractor` implements the `FeatureExtractor` Protocol for the `market`
family. It reads bookmaker pricing for the current match from
`OddsSnapshotRepository`, anchored on Pinnacle. The devig method is FIXED per
feature key by the §15.5 catalog (Shin for the `p1_implied_pinnacle_*` keys,
proportional for `p1_implied_proportional_decision`) — pinned as structural
constants, NOT read from `config.features.market` (see `_SHIN`/`_PROPORTIONAL`):

  - `p1_implied_pinnacle_opening` — `opening()` Pinnacle/Shin `p1_implied`.
  - `p1_implied_pinnacle_closing` — `closing()` Pinnacle/Shin `p1_implied`.
    **Backtest-only (§M19): NULL unless `fctx.match.status == 'final'`.**
  - `p1_implied_pinnacle_decision` — latest Pinnacle/Shin snapshot with
    `captured_at <= as_of_ts` (`latest_before`). This is the live decisioning value.
  - `p1_implied_proportional_decision` — same instant, proportional devig (cross-check).
  - `line_movement_p1` — `closing - opening` (Pinnacle/Shin). **Backtest-only (§M19).**
  - `consensus_implied_p1` — cross-bookmaker mean of decision-time `p1_implied`
    (Shin) over `config.sources.odds_api.bookmakers` (Pinnacle + Betfair exchanges).
  - `vig_pinnacle_decision` — `vig` of the decision-time Pinnacle/Shin snapshot.
  - `odds_drift_to_close` — per-tournament average of `|line_movement|`.
    **DEFERRED in v1: always NULL** (a cross-match tournament aggregate with no repo
    support yet — mirrors the §M3 interaction deferral). The key is still seeded for
    §M7 lockstep; populating it is a future session (and would need a DECISIONS.md
    update). Backtest-only when implemented.

§M19 (live-vs-backtest gating): the *mode* discriminator is `fctx.match.status`,
NOT `as_of_ts`. C4 fixes `status` as the mode signal (`for_training()` → `final`;
`for_prediction()` → `scheduled`/`live`), and a historical training row also has
`as_of_ts < start_ts`, so `as_of_ts` cannot distinguish backtest from live. A closing
line is a future fact at a live decision instant, so the closing-derived keys
(`*_closing`, `line_movement_p1`, `odds_drift_to_close`) are emitted only for a
`final` match; otherwise they are NULL (no look-ahead leak).

Every key is non-critical (§0.5/§M8) and odds coverage starts ~2020 (H11). C9
(`allow_missing_odds`): when no Pinnacle snapshot exists, every key is NULL —
never a hard failure. A `StorageError` from the repo degrades the affected read to
NULL (logged, redacted); a genuine programming defect propagates to the agent's
per-match dead-letter (§M16 / R6b M1 idiom).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.errors import StorageError
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import OddsSnapshotRepository
from tennis.storage.postgres.rows import DevigMethod, OddsSnapshotRow
from tennis.agents.research.context import FeatureContext

_logger = get_logger("tennis.agents.research.features.market")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
MARKET_FAMILY = "market"

# The devig method is FIXED per feature key by the §15.5 catalog (NOT config): the
# `p1_implied_pinnacle_*` keys are Shin, `p1_implied_proportional_decision` is
# proportional. Pinning them as structural constants (mirrors conditions.py's
# `_SOURCE = "owm"`) prevents a `config.features.market` primary/fallback swap from
# silently storing proportional values under Shin-labelled keys (Codex R7 MEDIUM):
# the key NAME encodes the method, so the method cannot be config-driven. The
# `config.features.market` devig knobs govern the devig PIPELINE, not these
# catalog-fixed per-key methods.
_SHIN: DevigMethod = "shin"
_PROPORTIONAL: DevigMethod = "proportional"

# The anchor bookmaker for the Pinnacle-specific keys. A market-source literal (it
# matches how the Odds API adapter writes `odds_snapshots.bookmaker`), NOT a config
# key — mirrors conditions.py's `_SOURCE = "owm"`. The cross-book consensus iterates
# the full configured bookmaker list (which includes this anchor).
_PINNACLE = "pinnacle"

# The status that marks a backtest/training row (C4). Only a `final` match may emit
# the closing-derived keys (§M19); scheduled/live/cancelled NULL them.
_BACKTEST_STATUS = "final"

# The 8 keys this family emits (§15.5 `features.market`, in catalog order). Must stay
# in lockstep with the `"market"` rows seeded by specs.py — guarded by the round-trip
# test (§M7).
MARKET_FEATURE_KEYS: tuple[str, ...] = (
    "p1_implied_pinnacle_opening",
    "p1_implied_pinnacle_closing",
    "p1_implied_pinnacle_decision",
    "p1_implied_proportional_decision",
    "line_movement_p1",
    "consensus_implied_p1",
    "vig_pinnacle_decision",
    "odds_drift_to_close",
)


class MarketExtractor:
    """`FeatureExtractor` for the `market` family. Stateless given its injected
    `OddsSnapshotRepository` and config."""

    name = MARKET_FAMILY

    def __init__(
        self,
        *,
        odds_repo: OddsSnapshotRepository,
        config: AppConfig,
    ) -> None:
        self._odds_repo = odds_repo
        self._bookmakers = tuple(config.sources.odds_api.bookmakers)

    def feature_keys(self) -> tuple[str, ...]:
        return MARKET_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        match_id = fctx.match.match_id
        as_of = fctx.as_of_ts
        # §M19: status — not as_of_ts — is the backtest/live discriminator.
        is_backtest = fctx.match.status == _BACKTEST_STATUS

        opening = self._safe(
            lambda: self._odds_repo.opening(
                match_id=match_id, bookmaker=_PINNACLE, devig_method=_SHIN
            ),
            match_id=match_id,
            what="opening",
        )
        decision = self._safe(
            lambda: self._odds_repo.latest_before(
                match_id=match_id,
                bookmaker=_PINNACLE,
                devig_method=_SHIN,
                captured_before=as_of,
            ),
            match_id=match_id,
            what="decision_shin",
        )
        decision_prop = self._safe(
            lambda: self._odds_repo.latest_before(
                match_id=match_id,
                bookmaker=_PINNACLE,
                devig_method=_PROPORTIONAL,
                captured_before=as_of,
            ),
            match_id=match_id,
            what="decision_proportional",
        )
        # Closing is a future fact for a live decision — read it only for a backtest
        # (final) row (§M19).
        closing = (
            self._safe(
                lambda: self._odds_repo.closing(
                    match_id=match_id, bookmaker=_PINNACLE, devig_method=_SHIN
                ),
                match_id=match_id,
                what="closing",
            )
            if is_backtest
            else None
        )

        opening_p1 = opening.p1_implied if opening is not None else None
        closing_p1 = closing.p1_implied if closing is not None else None
        line_movement = (
            closing_p1 - opening_p1
            if is_backtest and opening_p1 is not None and closing_p1 is not None
            else None
        )

        return {
            "p1_implied_pinnacle_opening": opening_p1,
            "p1_implied_pinnacle_closing": closing_p1,
            "p1_implied_pinnacle_decision": (
                decision.p1_implied if decision is not None else None
            ),
            "p1_implied_proportional_decision": (
                decision_prop.p1_implied if decision_prop is not None else None
            ),
            "line_movement_p1": line_movement,
            "consensus_implied_p1": self._consensus(match_id, as_of),
            "vig_pinnacle_decision": (
                decision.vig if decision is not None else None
            ),
            # DEFERRED (v1): per-tournament |line_movement| average — a cross-match
            # aggregate with no repo support. Always NULL; populating it needs a
            # future session + a DECISIONS.md update.
            "odds_drift_to_close": None,
        }

    def _consensus(self, match_id: int, as_of: datetime) -> float | None:
        """Cross-bookmaker mean of decision-time `p1_implied` (Shin devig) over the
        configured bookmakers. NULL when no book has a qualifying snapshot."""
        implied: list[float] = []
        for bookmaker in self._bookmakers:
            row = self._safe(
                lambda bk=bookmaker: self._odds_repo.latest_before(
                    match_id=match_id,
                    bookmaker=bk,
                    devig_method=_SHIN,
                    captured_before=as_of,
                ),
                match_id=match_id,
                what=f"consensus_{bookmaker}",
            )
            if row is not None:
                implied.append(row.p1_implied)
        if not implied:
            return None
        return sum(implied) / len(implied)

    def _safe(
        self,
        call: Callable[[], OddsSnapshotRow | None],
        *,
        match_id: int,
        what: str,
    ) -> OddsSnapshotRow | None:
        """Run one repo read; a `StorageError` degrades to None (→ NULL feature, C9)
        rather than raising. Only the repo's typed DB/IO failure is caught — a real
        defect (TypeError/AttributeError/…) propagates to the agent's per-match
        isolation, which dead-letters loudly (R6b M1 idiom). Logged, redacted (§L10)."""
        try:
            return call()
        except StorageError as exc:
            _logger.warning(
                "market_read_failed",
                match_id=match_id,
                what=what,
                cause=f"{type(exc).__name__}: {redact_text(str(exc))}",
            )
            return None
