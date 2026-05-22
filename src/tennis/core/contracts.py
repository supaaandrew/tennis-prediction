"""Cross-module Protocols.

Anything that crosses a module boundary goes through a Protocol defined here.
Concrete implementations live in `adapters/`, `storage/`, `agents/`. The rest
of the project depends only on these abstract shapes — that's what makes
testing and dependency swapping trivial.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from tennis.core.clock import Clock  # re-exported below for ergonomics
from tennis.core.lineage import (
    AgentLineage,
    HeartbeatPolicy,
    Precondition,
    RunStatus,
)


# ---------------------------------------------------------------------------
# Generic HTTP
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        import json
        return json.loads(self.body.decode("utf-8"))


@runtime_checkable
class HttpClient(Protocol):
    """Minimal HTTP surface. All adapters depend on this, not httpx directly."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> HttpResponse: ...

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> HttpResponse: ...


# ---------------------------------------------------------------------------
# Agent framework — the Pipeline talks to these, not concrete agents
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str
    cause: str | None = None
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentResult:
    ok: bool
    metrics: Mapping[str, Any]
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything an agent needs at runtime. Passed by the orchestrator.

    The agent never reaches into globals — config, clock, db, logger come
    from here. This is what makes tests and replay runs possible.
    """

    run_id: UUID
    as_of: datetime
    config: Any         # AppConfig — typed weakly here to keep contracts.py import-light
    db: Any             # SessionFactory protocol (defined in storage/postgres)
    clock: Clock
    logger: Any         # structlog BoundLogger


@runtime_checkable
class Agent(Protocol):
    """An agent has a name, a lineage declaration, and a single `run`
    entrypoint. The orchestrator inspects `lineage.preconditions` before
    invoking `run`; a missing precondition raises PreconditionNotMetError.
    """

    name: str
    lineage: AgentLineage

    def run(self, ctx: AgentContext) -> AgentResult: ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    data: Mapping[str, Any]
    error: AgentError | None = None


@runtime_checkable
class Tool(Protocol):
    """Tools are the atomic units an agent composes.

    Each tool is dispatch + a single side effect (write, fetch, compute).
    No agent business logic lives inside a tool.
    """

    name: str

    def __call__(self, ctx: AgentContext, /, **kwargs: Any) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------
__all__ = [
    "Clock",
    "HttpResponse",
    "HttpClient",
    "AgentError",
    "AgentResult",
    "AgentContext",
    "Agent",
    "AgentLineage",
    "HeartbeatPolicy",
    "Precondition",
    "RunStatus",
    "ToolResult",
    "Tool",
]
