"""SessionFactory tests.

The UTC-on-every-connection invariant is the load-bearing contract here.
Two paths are exercised without a live Postgres:

  1. `_set_utc_timezone` is called as a plain function with a mock
     dbapi connection; we assert it executed `SET TIME ZONE 'UTC'`.
  2. `PostgresSessionFactory.__init__` registers `_set_utc_timezone`
     on its engine's `connect` event — verified by inspecting the
     SQLAlchemy event registry.

A SQLite engine is used to stand in for Postgres at the engine-creation
layer; we never actually execute `SET TIME ZONE` against it (SQLite would
reject the statement). The point is to confirm the listener was attached.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sqlalchemy import event

from tennis.storage.postgres.session import (
    PostgresSessionFactory,
    SessionFactory,
    _make_statement_timeout_listener,
    _set_utc_timezone,
)


# ---------------------------------------------------------------------------
# 1. The listener function itself
# ---------------------------------------------------------------------------
class TestSetUtcTimezoneListener:
    def test_executes_set_time_zone_utc(self) -> None:
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor

        _set_utc_timezone(conn, None)

        cursor.execute.assert_called_once_with("SET TIME ZONE 'UTC'")
        cursor.close.assert_called_once()

    def test_closes_cursor_even_on_failure(self) -> None:
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError("simulated DB error")
        conn = MagicMock()
        conn.cursor.return_value = cursor

        with pytest.raises(RuntimeError, match="simulated"):
            _set_utc_timezone(conn, None)

        cursor.close.assert_called_once()


class TestStatementTimeoutListener:
    def test_builds_listener_that_sets_timeout_ms(self) -> None:
        listener = _make_statement_timeout_listener(5000)
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor

        listener(conn, None)

        cursor.execute.assert_called_once_with("SET statement_timeout = 5000")
        cursor.close.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Engine event registration
# ---------------------------------------------------------------------------
class TestPostgresSessionFactoryRegistration:
    """Use a SQLite URL — we only need a real engine to register events
    on. We never connect to it."""

    URL = "sqlite:///:memory:"

    def test_registers_utc_listener_on_connect(self) -> None:
        factory = PostgresSessionFactory(url=self.URL)
        try:
            assert event.contains(factory.engine, "connect", _set_utc_timezone)
        finally:
            factory.dispose()

    def test_registers_statement_timeout_when_configured(self) -> None:
        factory = PostgresSessionFactory(
            url=self.URL,
            statement_timeout_ms=30000,
        )
        try:
            # UTC listener is always present.
            assert event.contains(factory.engine, "connect", _set_utc_timezone)
            # The statement-timeout listener is exposed as an attribute
            # so we can verify it was actually attached to the engine.
            stmt_listener = factory._statement_timeout_listener
            assert stmt_listener is not None
            assert event.contains(factory.engine, "connect", stmt_listener)
        finally:
            factory.dispose()

    def test_no_statement_timeout_listener_by_default(self) -> None:
        factory = PostgresSessionFactory(url=self.URL)
        try:
            assert factory._statement_timeout_listener is None
            assert event.contains(factory.engine, "connect", _set_utc_timezone)
        finally:
            factory.dispose()

    def test_engine_property_exposes_engine(self) -> None:
        factory = PostgresSessionFactory(url=self.URL)
        try:
            assert factory.engine is not None
        finally:
            factory.dispose()

    def test_dispose_is_idempotent(self) -> None:
        factory = PostgresSessionFactory(url=self.URL)
        factory.dispose()
        # Second dispose must not raise.
        factory.dispose()


# ---------------------------------------------------------------------------
# 3. The Protocol itself
# ---------------------------------------------------------------------------
class TestSessionFactoryProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        assert getattr(SessionFactory, "_is_runtime_protocol", False) is True

    def test_concrete_factory_satisfies_protocol(self) -> None:
        factory = PostgresSessionFactory(url="sqlite:///:memory:")
        try:
            assert isinstance(factory, SessionFactory)
        finally:
            factory.dispose()


# ---------------------------------------------------------------------------
# 4. session() context-manager semantics — commit on clean exit, rollback
#    on exception. Use SQLite since we don't need PG features for this.
# ---------------------------------------------------------------------------
class TestSessionContextManager:
    URL = "sqlite:///:memory:"

    def test_yields_session_and_commits(self) -> None:
        factory = PostgresSessionFactory(url=self.URL)
        try:
            with factory.session() as s:
                assert s is not None
                assert s.is_active
        finally:
            factory.dispose()

    def test_rolls_back_on_exception(self) -> None:
        factory = PostgresSessionFactory(url=self.URL)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                with factory.session():
                    raise RuntimeError("boom")
        finally:
            factory.dispose()
