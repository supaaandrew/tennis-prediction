"""Postgres storage subpackage.

Package-level re-exports cover only the SIDE-EFFECT-FREE surface — Row
dataclasses and Repository Protocols. The SQLAlchemy-backed `session`
module is NOT eagerly imported here; consumers that need the session
factory import it explicitly:

    from tennis.storage.postgres.session import (
        SessionFactory, PostgresSessionFactory,
    )

This keeps the rows/protocol surface importable without pulling
SQLAlchemy into every consumer (a constraint flagged by the adversarial
review). ORM models (`models.py`) are similarly opt-in — only concrete
repository implementations import them.
"""

from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    EloSnapshotRepository,
    FeatureMatrixRepository,
    FeatureSpecRepository,
    IngestWatermarkRepository,
    MatchRepository,
    MatchStatRepository,
    ModelRegistryRepository,
    OddsSnapshotRepository,
    PipelineRunRepository,
    PlayerAliasRepository,
    PlayerRankingRepository,
    PlayerRepository,
    PredictionRepository,
    TournamentRepository,
    VenueRepository,
    WeatherObservationRepository,
    WeatherRevisionRepository,
)
from tennis.storage.postgres.rows import (
    Backhand,
    Confidence,
    DeadLetterRow,
    DevigMethod,
    DominantHand,
    Dtype,
    EloSnapshotRow,
    EloSurface,
    FeatureMatrixRow,
    FeatureSpecRow,
    IngestWatermarkRow,
    MatchDateSource,
    MatchRow,
    MatchStatRow,
    MatchStatus,
    ModelRegistryRow,
    OddsSnapshotRow,
    Perspective,
    PipelineRunRow,
    PlayerAliasRow,
    PlayerRankingRow,
    PlayerRow,
    PredictionRow,
    Round,
    RunStatusLit,
    Surface,
    Tier,
    TournamentRow,
    VenueRow,
    WeatherObservationRow,
    WeatherRevisionRow,
)

__all__ = [
    # Rows
    "PlayerRow",
    "PlayerRankingRow",
    "VenueRow",
    "TournamentRow",
    "MatchRow",
    "MatchStatRow",
    "OddsSnapshotRow",
    "WeatherObservationRow",
    "WeatherRevisionRow",
    "FeatureSpecRow",
    "FeatureMatrixRow",
    "ModelRegistryRow",
    "PredictionRow",
    "PipelineRunRow",
    "IngestWatermarkRow",
    "DeadLetterRow",
    "PlayerAliasRow",
    "EloSnapshotRow",
    # Enum aliases
    "Surface",
    "EloSurface",
    "Tier",
    "Round",
    "MatchStatus",
    "DominantHand",
    "Backhand",
    "Dtype",
    "Confidence",
    "RunStatusLit",
    "DevigMethod",
    "MatchDateSource",
    "Perspective",
    # Repository Protocols
    "PlayerRepository",
    "PlayerRankingRepository",
    "VenueRepository",
    "TournamentRepository",
    "MatchRepository",
    "MatchStatRepository",
    "OddsSnapshotRepository",
    "WeatherObservationRepository",
    "WeatherRevisionRepository",
    "FeatureSpecRepository",
    "FeatureMatrixRepository",
    "ModelRegistryRepository",
    "PredictionRepository",
    "PipelineRunRepository",
    "IngestWatermarkRepository",
    "DeadLetterRepository",
    "PlayerAliasRepository",
    "EloSnapshotRepository",
]
