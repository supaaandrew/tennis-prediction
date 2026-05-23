"""SQLAlchemy session factory with UTC-locked connections.

Why an event listener and not just a startup `SET TIME ZONE`:
  Connection pools recreate physical connections over the lifetime of the
  process (after timeouts, after errors, after pool recycle). A single
  startup statement runs on the connection that exists *at that moment*
  and does nothing for the next 200 connections the pool will spin up.
  The `connect` event fires for every new physical connection, so the
  invariant holds for the entire process.

The H1/B2 bug class (`now() AT TIME ZONE 'UTC'` returning naive timestamps
that get reinterpreted through the session TZ) is fully neutralized as
long as the session TZ is locked to UTC on every connection.

This module is the only place the project couples SQLAlchemy to Postgres
at runtime; the rest depends on `repositories.py` Protocols.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import event
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


# ---------------------------------------------------------------------------
# Module-level listener — picked up by `event.listen(engine, "connect", ...)`.
# Pulled out as a free function so unit tests can call it with a mock
# dbapi connection (no live DB required).
# ---------------------------------------------------------------------------
def _set_utc_timezone(dbapi_connection: Any, _connection_record: Any) -> None:
    """Force session TZ to UTC on a freshly-checked-out connection."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET TIME ZONE 'UTC'")
    finally:
        cursor.close()


def _make_statement_timeout_listener(timeout_ms: int) -> Any:
    """Build a `connect` listener that sets `statement_timeout` (ms)."""

    def _set_statement_timeout(
        dbapi_connection: Any, _connection_record: Any
    ) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"SET statement_timeout = {int(timeout_ms)}")
        finally:
            cursor.close()

    return _set_statement_timeout


# ---------------------------------------------------------------------------
# Protocol — agents depend on this, not on the concrete factory below.
# ---------------------------------------------------------------------------
@runtime_checkable
class SessionFactory(Protocol):
    """Hands out SQLAlchemy sessions bound to a UTC-locked engine."""

    def session(self) -> Any:  # contextmanager-returning method
        """Yield a Session inside a transactional block. The context
        manager commits on clean exit and rolls back on exception.
        """
        ...

    def dispose(self) -> None:
        """Tear down the underlying engine + pool. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Concrete factory
# ---------------------------------------------------------------------------
class PostgresSessionFactory:
    """Build a Postgres engine, attach the UTC listener (and optional
    statement-timeout listener), and yield contextual sessions.

    Constructed once per process by the DI container at startup.
    """

    def __init__(
        self,
        *,
        url: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        statement_timeout_ms: int | None = None,
    ) -> None:
        # QueuePool args (pool_size, max_overflow) are rejected by dialects
        # that use SingletonThreadPool / StaticPool (e.g. sqlite). We rely
        # on these only for the real Postgres pool; for any other dialect
        # we let SQLAlchemy pick its default pool class.
        engine_kwargs: dict[str, Any] = {"future": True}
        if url.startswith(("postgresql", "postgres")):
            engine_kwargs["pool_size"] = pool_size
            engine_kwargs["max_overflow"] = max_overflow
        self._engine: Engine = create_engine(url, **engine_kwargs)
        # CRITICAL: fires for every new physical connection, not just startup.
        event.listen(self._engine, "connect", _set_utc_timezone)
        # Held as an attribute so callers / tests can introspect (and so the
        # listener isn't garbage-collected before the engine).
        self._statement_timeout_listener: Any | None = None
        if statement_timeout_ms is not None:
            self._statement_timeout_listener = _make_statement_timeout_listener(
                statement_timeout_ms
            )
            event.listen(self._engine, "connect", self._statement_timeout_listener)
        self._sessionmaker = sessionmaker(self._engine, expire_on_commit=False)

    @property
    def engine(self) -> Engine:
        """Exposed for migrations / introspection. Repositories should NOT
        reach into the engine; use `session()` instead.
        """
        return self._engine

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Open a session, commit on clean exit, rollback on exception."""
        s = self._sessionmaker()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def dispose(self) -> None:
        self._engine.dispose()
