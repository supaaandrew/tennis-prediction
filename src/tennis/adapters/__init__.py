"""External-source adapters.

Each adapter reads from one upstream source (Sackmann mirror, ATP scraper,
OpenWeatherMap, The Odds API), parses raw payloads into Row DTOs, resolves
cross-source identities, and writes through the storage-layer repository
Protocols. Adapters never touch SQLAlchemy directly — they depend only on
the abstract repositories from `tennis.storage.postgres.repositories`.
"""
