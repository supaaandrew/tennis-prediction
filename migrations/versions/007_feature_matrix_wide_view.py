"""Feature matrix wide-column materialization - STUB.

This migration is intentionally a no-op for now.

Why it exists:
    feature_matrix.payload is JSONB, which is flexible but loses column
    statistics and is slow to scan once the feature count grows. When
    feature count exceeds ~500 (per architecture decision), this migration
    is where the wide materialization lives:
      - new table  feature_matrix_wide_v{N}  with one column per feature
      - populated by an offline rebuild job from feature_matrix.payload
      - reads in modeling/load_features.py switch to the wide table

Until that threshold is hit, the JSONB form is the source of truth.

Revision ID: 007
Revises: 006
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Intentional no-op. See module docstring for activation criteria.
    pass


def downgrade() -> None:
    pass
