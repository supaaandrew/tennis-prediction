"""Structured logging.

All logs are JSON in production: `{event, timestamp, level, agent_name, ...}`.
Console rendering is available for local dev. Context vars (run_id, agent,
match_id, etc.) are merged in automatically via structlog's contextvars
processor so business-logic call sites don't have to thread them through.

Redaction happens here, not at the call site — any key whose name matches
the configured redact list (case-insensitive) has its value replaced with
``"***"`` before serialization. Defense in depth against accidentally
logging a secret.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars
from structlog.stdlib import BoundLogger

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_contextvars",
    "clear_contextvars",
    "unbind_contextvars",
]


def _make_redactor(redact_keys: Sequence[str]) -> Any:
    """Build a structlog processor that masks values for sensitive keys.

    Matching is case-insensitive and matches by *substring* — `"api_key"`
    in the config list also masks `"openweather_api_key"`. The redactor
    walks recursively into mappings AND sequences (lists/tuples) so secrets
    nested inside a list of payload records still get masked. str/bytes are
    not treated as sequences (else we'd iterate characters).
    """
    needles = tuple(k.lower() for k in redact_keys)

    def _is_secret(key: str) -> bool:
        kl = key.lower()
        return any(needle in kl for needle in needles)

    def _mask(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: ("***" if _is_secret(str(k)) else _mask(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            masked = [_mask(item) for item in value]
            return masked if isinstance(value, list) else tuple(masked)
        return value

    def processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        for key in list(event_dict.keys()):
            if _is_secret(key):
                event_dict[key] = "***"
            else:
                event_dict[key] = _mask(event_dict[key])
        return event_dict

    return processor


def configure_logging(
    *,
    level: str = "INFO",
    json: bool = True,
    redact_keys: Sequence[str] = (),
) -> None:
    """Configure structlog + stdlib logging for the whole process.

    Safe to call multiple times — later calls replace earlier configuration,
    which is what tests want.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _make_redactor(redact_keys),
    ]

    renderer: Any
    if json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_context: Any) -> BoundLogger:
    """Return a structlog logger, optionally pre-bound with context fields."""
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger  # type: ignore[no-any-return]
