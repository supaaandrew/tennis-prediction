"""DI container tests — singleton vs factory, error paths."""

from __future__ import annotations

from typing import Protocol

import pytest

from tennis.core.di import Container, ContainerError


class Greeter(Protocol):
    def hello(self) -> str: ...


class _Hi:
    def hello(self) -> str:
        return "hi"


class TestContainer:
    def test_singleton_resolves_same_instance(self) -> None:
        c = Container()
        c.register_singleton(Greeter, _Hi)
        a = c.resolve(Greeter)
        b = c.resolve(Greeter)
        assert a is b

    def test_factory_resolves_fresh_instance(self) -> None:
        c = Container()
        c.register_factory(Greeter, _Hi)
        a = c.resolve(Greeter)
        b = c.resolve(Greeter)
        assert a is not b

    def test_double_registration_rejected(self) -> None:
        c = Container()
        c.register_singleton(Greeter, _Hi)
        with pytest.raises(ContainerError, match="already registered"):
            c.register_singleton(Greeter, _Hi)
        with pytest.raises(ContainerError, match="already registered"):
            c.register_factory(Greeter, _Hi)

    def test_resolve_unknown_raises(self) -> None:
        c = Container()
        with pytest.raises(ContainerError, match="No registration"):
            c.resolve(Greeter)

    def test_has_reports_presence(self) -> None:
        c = Container()
        assert not c.has(Greeter)
        c.register_factory(Greeter, _Hi)
        assert c.has(Greeter)

    def test_reset_clears(self) -> None:
        c = Container()
        c.register_singleton(Greeter, _Hi)
        c.resolve(Greeter)
        c.reset()
        assert not c.has(Greeter)
