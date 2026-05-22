"""Integration test fixtures.

Spins a real Postgres in a Docker container, runs every migration head-on,
and hands tests a live SQLAlchemy engine. If Docker / testcontainers /
alembic / psycopg / sqlalchemy aren't all available, every test that pulls
the `postgres_engine` fixture is skipped cleanly — there is no silent
'no DB so the test trivially passes' path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _docker_available() -> bool:
    """True iff a usable Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Any]:
    """A SQLAlchemy engine bound to a freshly-migrated Postgres container.

    Skips the test (not the suite) when any prerequisite is missing.
    """
    pytest.importorskip("testcontainers", reason="install testcontainers[postgres]")
    pytest.importorskip("alembic", reason="install alembic")
    pytest.importorskip("sqlalchemy", reason="install sqlalchemy")
    pytest.importorskip("psycopg", reason="install psycopg[binary]")

    if not _docker_available():
        pytest.skip("Docker daemon not reachable — skipping Postgres integration test")

    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig
    from sqlalchemy import create_engine, text
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer("postgres:16-alpine", driver="psycopg")
    container.start()
    try:
        url = container.get_connection_url()
        # Force UTC session TZ — mirrors what SessionFactory will do in prod.
        engine = create_engine(url, future=True)
        with engine.connect() as conn:
            conn.execute(text("SET TIME ZONE 'UTC'"))
            conn.commit()

        # Run every migration head-on against the container.
        os.environ["DATABASE_URL"] = url
        alembic_cfg = AlembicConfig(str(_REPO_ROOT / "ops" / "alembic.ini"))
        alembic_cfg.set_main_option(
            "script_location", str(_REPO_ROOT / "migrations")
        )
        alembic_command.upgrade(alembic_cfg, "head")

        yield engine
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
        container.stop()
