"""Point-in-time (PIT) cutoff — the single source of the `as_of_ts` stamp.

Load-bearing per DECISIONS §8 / §A14. Every feature row the Research Agent
writes is stamped with `as_of_ts = pit_cut(match)`; this is the instant at which
the feature is "known" and is what `FeatureMatrixValidator` rule R4 checks. The
Postgres `fm_no_lookahead` trigger is defense-in-depth only and is *more lenient*
than this rule — `pit_cut` is the authoritative cut.

Two states (§8):

  - live / scheduled (`start_ts IS NOT NULL`): `as_of_ts = start_ts - live_offset`
    where `live_offset = config.decision_timing.live_decision_offset_hours` (24).
  - historical Sackmann (`start_ts IS NULL`): `as_of_ts = match_date - 1 day` at
    00:00 UTC — conservative; discards same-day morning info (§3.2).

Pure functions over `(MatchRow, ...)`: no IO, no clock (the cut is derived from
the match's own fields), always tz-aware UTC output.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from tennis.storage.postgres.rows import MatchRow

# The historical cut subtracts a full calendar day. This is a STRUCTURAL PIT
# constant (§8 / §A14 / §3.2) — the conservative "match_date − 1 day" rule — and
# is intentionally NOT a tunable config threshold. The live offset, by contrast,
# IS config-backed and is passed in by the caller.
_HISTORICAL_PIT_OFFSET_DAYS = 1


def pit_cut(match: MatchRow, *, live_offset_hours: int) -> datetime:
    """Return the point-in-time `as_of_ts` for `match` as a tz-aware UTC instant.

    `live_offset_hours` is the caller-supplied live decision offset (M-b:
    `config.decision_timing.live_decision_offset_hours`, never hardcoded). It
    must be positive: a zero or negative offset would yield `as_of_ts >=
    start_ts` — a lookahead that every live row would (correctly) fail R4 on, so
    we reject it here rather than silently emit invalid rows.
    """
    if live_offset_hours <= 0:
        raise ValueError(
            f"live_offset_hours must be positive (got {live_offset_hours}); "
            "a non-positive offset yields as_of_ts >= start_ts (lookahead)"
        )
    if match.start_ts is not None:
        # live / scheduled — cut `live_offset_hours` before the start instant.
        # `start_ts` is contractually tz-aware UTC; reject a naive value loudly
        # (a blind astimezone would reinterpret it as local time and silently
        # produce the wrong instant), then normalize to honor the UTC contract.
        if match.start_ts.tzinfo is None:
            raise ValueError("pit_cut: match.start_ts must be timezone-aware")
        return (match.start_ts - timedelta(hours=live_offset_hours)).astimezone(UTC)
    # historical Sackmann — 00:00 UTC of the day before `match_date`.
    cut_date = match.match_date - timedelta(days=_HISTORICAL_PIT_OFFSET_DAYS)
    return datetime.combine(cut_date, time.min, tzinfo=UTC)
