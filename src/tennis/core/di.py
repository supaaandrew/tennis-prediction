"""Tiny typed dependency injection container.

Keeps wiring centralized — adapters, repositories, agents are registered once
at startup (in `tennis.cli` or the orchestrator) and resolved by Protocol type
elsewhere. No magic, no scanning. Two scopes: `singleton` (instantiated once)
and `factory` (new instance each resolve).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tennis.core.errors import TennisError

T = TypeVar("T")


class ContainerError(TennisError):
    """A container registration or resolution failed."""


class Container:
    """Pythonic DI: register factories keyed by Protocol/ABC, resolve by type."""

    def __init__(self) -> None:
        self._singletons: dict[type, Callable[[], Any]] = {}
        self._factories: dict[type, Callable[[], Any]] = {}
        self._instances: dict[type, Any] = {}

    def register_singleton(self, key: type[T], factory: Callable[[], T]) -> None:
        if key in self._singletons or key in self._factories:
            raise ContainerError(f"{key!r} already registered")
        self._singletons[key] = factory

    def register_factory(self, key: type[T], factory: Callable[[], T]) -> None:
        if key in self._singletons or key in self._factories:
            raise ContainerError(f"{key!r} already registered")
        self._factories[key] = factory

    def resolve(self, key: type[T]) -> T:
        if key in self._instances:
            return self._instances[key]  # type: ignore[no-any-return]
        if key in self._singletons:
            obj = self._singletons[key]()
            self._instances[key] = obj
            return obj  # type: ignore[no-any-return]
        if key in self._factories:
            return self._factories[key]()  # type: ignore[no-any-return]
        raise ContainerError(f"No registration for {key!r}")

    def has(self, key: type) -> bool:
        return key in self._singletons or key in self._factories or key in self._instances

    def reset(self) -> None:
        """Forget all registrations. Mostly useful for tests."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()
