"""Feature extractor families for the Research Agent.

Each family is one module exporting a `FeatureExtractor` (see `base.py`):
Elo (R3), rankings/form/H2H (R4), serve/surface/weather (R5), fatigue/market
(R7). R2 ships only the `FeatureExtractor` Protocol; no family is implemented
yet (lockstep — see `agents/research/specs.py`).
"""
