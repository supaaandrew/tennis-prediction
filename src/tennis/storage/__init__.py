"""Storage layer.

`postgres/` holds the Postgres-backed implementation (rows, models, session,
repositories). Other backends would live as sibling packages. The rest of
the project imports either Row dataclasses or Repository Protocols — never
SQLAlchemy types directly.
"""
