"""Research Agent package. Exposes the `ResearchAgent` orchestrator (R6a) plus
the `FeatureMatrixValidator` the pipeline runs as the Research→Modeling gate.
"""

from tennis.agents.research.agent import ResearchAgent, ResearchMode
from tennis.agents.research.validator import (
    FeatureMatrixValidator,
    FeatureSpec,
)

__all__ = ["FeatureMatrixValidator", "FeatureSpec", "ResearchAgent", "ResearchMode"]
