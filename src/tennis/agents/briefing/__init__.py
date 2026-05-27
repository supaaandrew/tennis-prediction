"""Briefing Agent package (B1)."""

from __future__ import annotations

from tennis.agents.briefing.agent import BriefingAgent
from tennis.agents.briefing.email_client import EmailClient, SmtpEmailClient
from tennis.agents.briefing.llm_client import AnthropicLlmClient, LlmClient
from tennis.agents.briefing.render import SurfacedMatch, render_email

__all__ = [
    "BriefingAgent",
    "LlmClient",
    "AnthropicLlmClient",
    "EmailClient",
    "SmtpEmailClient",
    "SurfacedMatch",
    "render_email",
]
