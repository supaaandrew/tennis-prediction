"""`feature_specs` registry + seeding + the validator `expected_specs` builder.

This is the **lockstep** mechanism between the feature catalog and the
extractors: the registry tracks only the families whose extractors actually
exist, and the validator's `expected_specs` is built from the *registered*
families only. Seeding the full v1 catalog before its extractors exist would make
`FeatureMatrixValidator` reject every row for a "missing required feature".

Each later session appends its family's `FeatureSpecRow`s to `_REGISTRY`:

    _REGISTRY = {"elo": (FeatureSpecRow("p1_elo_pre", 1, "float"), ...)}   # R3

R2 ships the scaffold with no families registered (no extractor exists yet); the
mechanism is exercised in tests via an injected `registry` argument.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from tennis.core.errors import FeatureContractError
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import FeatureSpecRepository
from tennis.storage.postgres.rows import FeatureSpecRow
from tennis.agents.research.validator import _CRITICAL_FEATURE_KEYS, FeatureSpec

_logger = get_logger("tennis.agents.research.specs")

# family name -> the catalog rows that family emits.
FeatureSpecRegistry = Mapping[str, tuple[FeatureSpecRow, ...]]

# Canonical v1 Form windows (§15.5 / config.features.windows_days). The catalog
# is a static artifact, so the windows are pinned here; the round-trip test asserts
# `FormExtractor.feature_keys()` (built from config) equals these seeded rows, so a
# config/catalog divergence fails loud (§M7) rather than silently drifting.
_FORM_WINDOWS: tuple[int, ...] = (7, 14, 30, 90, 365)


def _form_rows() -> tuple[FeatureSpecRow, ...]:
    """The 25 §15.5 `features.form` rows: per window, win_rate (float) +
    matches_played (int) for each side, plus the win_rate_diff (float)."""
    rows: list[FeatureSpecRow] = []
    for w in _FORM_WINDOWS:
        rows.append(FeatureSpecRow(f"p1_win_rate_{w}d", 1, "float"))
        rows.append(FeatureSpecRow(f"p2_win_rate_{w}d", 1, "float"))
        rows.append(FeatureSpecRow(f"p1_matches_played_{w}d", 1, "int"))
        rows.append(FeatureSpecRow(f"p2_matches_played_{w}d", 1, "int"))
        rows.append(FeatureSpecRow(f"win_rate_diff_{w}d", 1, "float"))
    return tuple(rows)


# R3 landed the first family ("elo"); R4 appends "rankings", "form", "h2h";
# R5a appends "surface", "serve_return"; R5b appends "conditions"; R7 appends
# "fatigue", "market" — all in lockstep with their extractors. Each family's
# extractor `feature_keys()` MUST equal its seeded rows here — guarded by the
# per-family round-trip tests under `tests/unit/agents/research/features/`. The 9
# Elo keys mirror §15.5 (the 7 base ratings are the §M8 critical keys; the two
# reliability booleans are non-critical). The Rankings family is NEW to §15.5
# (§M11); Form/H2H mirror their §15.5 rows (H2H-advanced reconciled to §M1).
# Surface mirrors the locked 7-key §15.5 catalog (transition keys reconciled in
# §M13); serve_return mirrors its 15 §15.5 rows (§M14). Conditions mirrors the 9
# BASE §15.5 keys (§M15; the two §M3 interaction keys remain deferred — not in the
# v1 registry). All of R4/R5a/R5b's keys are non-critical (§0.5/§M8) — none enters
# `_CRITICAL_FEATURE_KEYS`.
_REGISTRY: dict[str, tuple[FeatureSpecRow, ...]] = {
    "elo": (
        FeatureSpecRow("p1_elo_pre", 1, "float"),
        FeatureSpecRow("p2_elo_pre", 1, "float"),
        FeatureSpecRow("p1_elo_surface_pre", 1, "float"),
        FeatureSpecRow("p2_elo_surface_pre", 1, "float"),
        FeatureSpecRow("p1_elo_blended_pre", 1, "float"),
        FeatureSpecRow("p2_elo_blended_pre", 1, "float"),
        FeatureSpecRow("elo_diff_blended", 1, "float"),
        FeatureSpecRow("p1_elo_reliability_low", 1, "bool"),
        FeatureSpecRow("p2_elo_reliability_low", 1, "bool"),
    ),
    "rankings": (
        FeatureSpecRow("p1_rank_pre", 1, "int"),
        FeatureSpecRow("p2_rank_pre", 1, "int"),
        FeatureSpecRow("rank_diff", 1, "int"),
        FeatureSpecRow("p1_rank_stale", 1, "bool"),
        FeatureSpecRow("p2_rank_stale", 1, "bool"),
    ),
    "form": _form_rows(),
    "h2h": (
        FeatureSpecRow("h2h_matches", 1, "int"),
        FeatureSpecRow("h2h_p1_wins", 1, "int"),
        FeatureSpecRow("h2h_p1_win_rate", 1, "float"),
        FeatureSpecRow("h2h_surface_matches", 1, "int"),
        FeatureSpecRow("h2h_surface_p1_win_rate", 1, "float"),
        FeatureSpecRow("h2h_win_rate_confidence", 1, "float"),
        FeatureSpecRow("h2h_win_rate_weighted", 1, "float"),
    ),
    "surface": (
        FeatureSpecRow("p1_career_win_rate_surface", 1, "float"),
        FeatureSpecRow("p2_career_win_rate_surface", 1, "float"),
        FeatureSpecRow("p1_recent_win_rate_surface_365d", 1, "float"),
        FeatureSpecRow("p2_recent_win_rate_surface_365d", 1, "float"),
        FeatureSpecRow("surface_affinity_diff", 1, "float"),
        FeatureSpecRow("surface_transition_type", 1, "cat"),
        FeatureSpecRow("surface_transition_exposure", 1, "float"),
    ),
    "serve_return": (
        FeatureSpecRow("p1_first_serve_pct_career", 1, "float"),
        FeatureSpecRow("p2_first_serve_pct_career", 1, "float"),
        FeatureSpecRow("p1_first_serve_pct_365d", 1, "float"),
        FeatureSpecRow("p2_first_serve_pct_365d", 1, "float"),
        FeatureSpecRow("p1_first_serve_win_pct_365d", 1, "float"),
        FeatureSpecRow("p2_first_serve_win_pct_365d", 1, "float"),
        FeatureSpecRow("p1_second_serve_win_pct_365d", 1, "float"),
        FeatureSpecRow("p2_second_serve_win_pct_365d", 1, "float"),
        FeatureSpecRow("p1_ace_rate_365d", 1, "float"),
        FeatureSpecRow("p2_ace_rate_365d", 1, "float"),
        FeatureSpecRow("p1_df_rate_365d", 1, "float"),
        FeatureSpecRow("p2_df_rate_365d", 1, "float"),
        FeatureSpecRow("p1_bp_save_pct_365d", 1, "float"),
        FeatureSpecRow("p2_bp_save_pct_365d", 1, "float"),
        FeatureSpecRow("serve_dominance_diff_365d", 1, "float"),
    ),
    "conditions": (
        FeatureSpecRow("temp_c_decision", 1, "float"),
        FeatureSpecRow("humidity_pct_decision", 1, "float"),
        FeatureSpecRow("wind_speed_ms_decision", 1, "float"),
        FeatureSpecRow("wind_dir_deg_decision", 1, "int"),
        FeatureSpecRow("precip_mm_decision", 1, "float"),
        FeatureSpecRow("cloud_pct_decision", 1, "int"),
        FeatureSpecRow("altitude_m", 1, "int"),
        FeatureSpecRow("indoor", 1, "bool"),
        FeatureSpecRow("forecast_uncertainty_bucket", 1, "cat"),
    ),
    # R7 — the two §15.5 families left unassigned by R2–R6. Both non-critical
    # (§M8). Fatigue: rest_days is whole days (int); the weighted counts/minutes and
    # the travel distance are floats. Market: all implied/movement/vig values are
    # floats; `odds_drift_to_close` is seeded for §M7 lockstep but emitted NULL in v1
    # (deferred — see features/market.py). Each family's extractor `feature_keys()`
    # MUST equal these rows (round-trip tests under tests/unit/.../features/).
    "fatigue": (
        FeatureSpecRow("p1_rest_days", 1, "int"),
        FeatureSpecRow("p2_rest_days", 1, "int"),
        FeatureSpecRow("p1_matches_last_7d", 1, "float"),
        FeatureSpecRow("p2_matches_last_7d", 1, "float"),
        FeatureSpecRow("p1_matches_last_14d", 1, "float"),
        FeatureSpecRow("p2_matches_last_14d", 1, "float"),
        FeatureSpecRow("p1_minutes_last_7d", 1, "float"),
        FeatureSpecRow("p2_minutes_last_7d", 1, "float"),
        FeatureSpecRow("p1_minutes_last_14d", 1, "float"),
        FeatureSpecRow("p2_minutes_last_14d", 1, "float"),
        FeatureSpecRow("p1_travel_km_since_last_match", 1, "float"),
        FeatureSpecRow("p2_travel_km_since_last_match", 1, "float"),
    ),
    "market": (
        FeatureSpecRow("p1_implied_pinnacle_opening", 1, "float"),
        FeatureSpecRow("p1_implied_pinnacle_closing", 1, "float"),
        FeatureSpecRow("p1_implied_pinnacle_decision", 1, "float"),
        FeatureSpecRow("p1_implied_proportional_decision", 1, "float"),
        FeatureSpecRow("line_movement_p1", 1, "float"),
        FeatureSpecRow("consensus_implied_p1", 1, "float"),
        FeatureSpecRow("vig_pinnacle_decision", 1, "float"),
        FeatureSpecRow("odds_drift_to_close", 1, "float"),
    ),
}


def seed_feature_specs(
    repo: FeatureSpecRepository,
    *,
    families: Iterable[str],
    registry: FeatureSpecRegistry = _REGISTRY,
) -> int:
    """Upsert the `FeatureSpecRow`s for the given (registered) families.

    Idempotent: `FeatureSpecRepository.upsert` is keyed on
    `(feature_key, version)`, so re-seeding leaves the catalog unchanged. An
    unregistered family name raises `KeyError` — loud by design; the caller must
    only seed families whose extractor exists. Returns the number of rows upserted.
    """
    family_list = list(families)  # materialize: `families` may be a one-shot iterator
    upserted = 0
    for family in family_list:
        for row in registry[family]:  # KeyError on unknown family — intentional
            repo.upsert(row)
            upserted += 1
    _logger.info("feature_specs_seeded", families=family_list, rows=upserted)
    return upserted


def build_expected_specs(
    repo: FeatureSpecRepository,
    *,
    feature_set: str,
    families: Iterable[str],
    registry: FeatureSpecRegistry = _REGISTRY,
) -> tuple[FeatureSpec, ...]:
    """Build the validator's `expected_specs` from the seeded catalog.

    Reads the active `FeatureSpecRow`s for `feature_set`, keeps only those
    belonging to the requested (registered) families, and lifts each into a
    `FeatureSpec`, stamping `critical` from `_CRITICAL_FEATURE_KEYS` (M-d: the
    `critical` flag is code-side, not a `feature_specs` column).
    """
    wanted_keys = {
        row.feature_key for family in families for row in registry[family]
    }
    expected = tuple(
        FeatureSpec(
            feature_key=row.feature_key,
            version=row.version,
            dtype=row.dtype,
            critical=row.feature_key in _CRITICAL_FEATURE_KEYS,
        )
        for row in repo.list_active(feature_set=feature_set)
        if row.feature_key in wanted_keys
    )
    # Catalog-drift guard: every registered key MUST be present in the seeded
    # catalog. A missing key (failed/partial seed, wrong migration/feature_set)
    # would otherwise be SILENTLY dropped from expected_specs — the validator
    # would then stop enforcing R1/R2/R3 for it, letting extractors and the
    # catalog diverge with no hard failure. Fail loud instead (§C10 ethos).
    missing = wanted_keys - {spec.feature_key for spec in expected}
    if missing:
        raise FeatureContractError(
            f"feature_specs catalog drift for feature_set={feature_set!r}: "
            f"registered keys not seeded: {sorted(missing)}"
        )
    return expected
