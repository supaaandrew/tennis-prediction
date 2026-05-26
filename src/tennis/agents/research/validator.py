"""Feature matrix validator — gate between Research and Modeling.

Called by `orchestrator/pipeline.py` AFTER the Research Agent finishes and
BEFORE the Modeling Agent starts. Deliberately not inside either agent —
the contract belongs at the seam, not on one side of it.

Rules enforced (all loud, never silent):

  R1. Every `FeatureSpec` is present in every row's payload.
  R2. Each present value's type matches the spec's `dtype`.
  R3. Critical fields (spec.critical=True) are non-null.
  R4. `row.as_of_ts < match.start_ts` for every row whose match has a
      `start_ts`. (Matches without a start_ts are checked against
      `match_date` interpreted as 00:00 UTC, mirroring the PIT trigger.)

All violations are collected and raised together as
`FeatureMatrixValidationError` so the operator sees every problem at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

from tennis.core.errors import FeatureMatrixValidationError

Dtype = Literal["float", "int", "bool", "cat"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One row from the `feature_specs` catalog, lifted into a value object."""

    feature_key: str
    version: int
    dtype: Dtype
    critical: bool = False


# Critical feature keys (DECISIONS §15.6 / mismatch M-d). This is deliberately
# NOT a `feature_specs` column — `FeatureSpecRow` has no `critical` field; the
# set lives here in code and the `expected_specs` builder stamps each
# `FeatureSpec(..., critical=key in _CRITICAL_FEATURE_KEYS)`.
#
# Kept MINIMAL per research_specs §0.5: only keys that are *never* legitimately
# NULL. The validator's `critical` flag is per-spec (global), not per-row, so it
# cannot express "critical only where coverage exists" — marking a sometimes-NULL
# key (form/serve/market/weather/ranking) critical would reject a legitimately
# sparse row (e.g. a debut player). The base Elo ratings always carry the 1500
# cold-start fallback (H10) and so are the only never-NULL keys today. Later
# sessions confirm/extend this set in lockstep with their extractors (R3 = Elo).
_CRITICAL_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        "elo_diff_blended",
        "p1_elo_pre",
        "p2_elo_pre",
        "p1_elo_surface_pre",
        "p2_elo_surface_pre",
        "p1_elo_blended_pre",
        "p2_elo_blended_pre",
    }
)


@dataclass(frozen=True, slots=True)
class _RowRef:
    """Compact reference to a row used in violation messages."""

    match_id: int
    feature_set: str

    def __str__(self) -> str:
        return f"(match_id={self.match_id}, feature_set={self.feature_set!r})"


@dataclass(frozen=True, slots=True)
class FeatureMatrixValidator:
    """Stateless validator. Construct with the expected feature specs and a
    match lookup; call `validate(rows=...)` with the rows about to be
    persisted.
    """

    expected_specs: tuple[FeatureSpec, ...]
    match_starts: Mapping[int, datetime | None]
    """match_id → start_ts (or None for historical rows)."""

    match_dates: Mapping[int, date]
    """match_id → match_date. Used for the PIT lower bound when start_ts is None."""

    def validate(self, *, rows: Iterable[Mapping[str, Any]]) -> None:
        """Iterate rows; raise FeatureMatrixValidationError with every
        violation found. Returns None on success.

        Each row is expected to carry at least:
            match_id     : int
            feature_set  : str
            as_of_ts     : datetime (tz-aware)
            payload      : Mapping[str, Any]
        """
        violations: list[str] = []
        for row in rows:
            ref = _RowRef(
                match_id=int(row["match_id"]),
                feature_set=str(row["feature_set"]),
            )
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                violations.append(f"{ref}: payload missing or not a mapping")
                continue
            self._check_payload(ref, payload, violations)
            self._check_pit(ref, row, violations)

        if violations:
            raise FeatureMatrixValidationError(violations=violations)

    # ------------------------------------------------------------------
    # Per-rule helpers
    # ------------------------------------------------------------------
    def _check_payload(
        self,
        ref: _RowRef,
        payload: Mapping[str, Any],
        violations: list[str],
    ) -> None:
        for spec in self.expected_specs:
            if spec.feature_key not in payload:
                # R1
                violations.append(
                    f"{ref}: missing required feature {spec.feature_key!r}"
                )
                continue
            value = payload[spec.feature_key]
            if value is None:
                # R3 — critical fields must not be null
                if spec.critical:
                    violations.append(
                        f"{ref}: critical feature {spec.feature_key!r} is null"
                    )
                continue
            # R2 — dtype
            if not _matches_dtype(value, spec.dtype):
                violations.append(
                    f"{ref}: feature {spec.feature_key!r} dtype mismatch "
                    f"(expected {spec.dtype}, got {type(value).__name__}={value!r})"
                )

    def _check_pit(
        self,
        ref: _RowRef,
        row: Mapping[str, Any],
        violations: list[str],
    ) -> None:
        as_of = row.get("as_of_ts")
        if not isinstance(as_of, datetime):
            violations.append(f"{ref}: as_of_ts missing or not a datetime")
            return
        if as_of.tzinfo is None:
            violations.append(f"{ref}: as_of_ts is naive (must be tz-aware UTC)")
            return

        start_ts = self.match_starts.get(ref.match_id, _MISSING)
        if start_ts is _MISSING:
            violations.append(
                f"{ref}: match_id not present in match_starts lookup"
            )
            return

        if start_ts is not None:
            assert isinstance(start_ts, datetime)
            if start_ts.tzinfo is None:
                violations.append(
                    f"{ref}: matches.start_ts is naive (must be tz-aware UTC)"
                )
                return
            if as_of >= start_ts:
                violations.append(
                    f"{ref}: as_of_ts {as_of.isoformat()} >= "
                    f"start_ts {start_ts.isoformat()} (PIT violation)"
                )
            return

        # Historical row — compare against match_date at UTC midnight.
        match_date = self.match_dates.get(ref.match_id)
        if match_date is None:
            violations.append(
                f"{ref}: match has neither start_ts nor match_date in lookups"
            )
            return
        midnight_utc = datetime(
            match_date.year, match_date.month, match_date.day, tzinfo=UTC
        )
        if as_of >= midnight_utc:
            violations.append(
                f"{ref}: as_of_ts {as_of.isoformat()} >= "
                f"match_date midnight UTC {midnight_utc.isoformat()} "
                f"(historical PIT violation)"
            )


# Sentinel distinct from None for "key absent vs key present-but-None".
_MISSING: object = object()


def _matches_dtype(value: Any, dtype: Dtype) -> bool:
    """True iff `value` matches the declared spec dtype.

    Notes:
      - bool is checked BEFORE int (bool is a subclass of int in Python).
      - `int` is acceptable where `float` is declared (an int is a float).
      - `cat` (categorical) is any non-None string.
    """
    if dtype == "bool":
        return isinstance(value, bool)
    if dtype == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if dtype == "cat":
        return isinstance(value, str)
    return False
