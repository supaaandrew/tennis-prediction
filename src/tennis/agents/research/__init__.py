"""Research Agent package. Foundation phase contains only the validator,
which the orchestrator runs as a gate between Research and Modeling.
"""

from tennis.agents.research.validator import (
    FeatureMatrixValidator,
    FeatureSpec,
)

__all__ = ["FeatureMatrixValidator", "FeatureSpec"]
