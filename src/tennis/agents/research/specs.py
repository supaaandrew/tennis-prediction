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

# Empty in R2 by design (lockstep — no extractor exists yet). R3+ append a family
# entry here in the same session that lands the family's extractor.
_REGISTRY: dict[str, tuple[FeatureSpecRow, ...]] = {}


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
