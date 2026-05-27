"""LLM client boundary for the Briefing Agent (§N2).

`LlmClient` is the `@runtime_checkable` Protocol the agent depends on — it never
imports `anthropic` directly (mirrors the `core/contracts.py` `HttpClient`
precedent). Unit tests inject a mock; the real `AnthropicLlmClient` is used only
in production. Any SDK/HTTP error is wrapped in a typed `BriefingLlmError` whose
message is `redact_text`'d (§L10) — a transport error can carry the API key.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tennis.core.config import LlmConfig
from tennis.core.errors import BriefingLlmError
from tennis.core.logging import get_logger, redact_text

_logger = get_logger("tennis.agents.briefing.llm_client")


@runtime_checkable
class LlmClient(Protocol):
    """Minimal narrative-generation surface. The agent depends on this shape."""

    def generate(self, *, prompt: str) -> str: ...


class AnthropicLlmClient:
    """Anthropic-backed `LlmClient`.

    The underlying `anthropic.Anthropic` client is injectable for testing
    (`client=`); in production it is built from `config.api_key_env` via
    `read_required_env` at construction. `generate` never lets a raw SDK
    exception escape — it wraps it in `BriefingLlmError` with a redacted cause.
    """

    def __init__(self, *, config: LlmConfig, client: Any | None = None) -> None:
        self._config = config
        if client is None:
            import anthropic  # local import — keep the SDK off the agent's import path

            from tennis.core.config import read_required_env

            client = anthropic.Anthropic(api_key=read_required_env(config.api_key_env))
        self._client = client

    def generate(self, *, prompt: str) -> str:
        try:
            resp = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_text(resp)
        except BriefingLlmError:
            raise
        except Exception as exc:  # SDK/HTTP/transport — wrap + redact (§L10)
            raise BriefingLlmError(redact_text(str(exc))) from exc


def _extract_text(resp: Any) -> str:
    """Concatenate the text blocks of an Anthropic Messages response."""
    blocks = getattr(resp, "content", None) or []
    parts = [getattr(b, "text", "") for b in blocks]
    text = "".join(p for p in parts if p).strip()
    if not text:
        raise BriefingLlmError("LLM returned an empty narrative")
    return text
