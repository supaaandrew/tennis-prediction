"""Concrete client tests for the Briefing Agent (B1, §N2 / §L10).

These exercise `AnthropicLlmClient` / `SmtpEmailClient` WITHOUT any network: the
SDK client and the SMTP transport are injected. They lock the error-wrapping +
`redact_text` contract and Protocol conformance.
"""

from __future__ import annotations

import pytest

from tennis.agents.briefing.email_client import EmailClient, SmtpEmailClient
from tennis.agents.briefing.llm_client import AnthropicLlmClient, LlmClient
from tennis.core.errors import BriefingEmailError, BriefingLlmError


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
class _Block:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, texts):
        self.content = [_Block(t) for t in texts]


class _FakeAnthropic:
    def __init__(self, *, resp=None, raise_exc=None):
        self._resp = resp
        self._raise = raise_exc
        self.calls = []
        self.messages = self  # messages.create lives on the same object

    def create(self, *, model, max_tokens, temperature, messages):
        self.calls.append((model, max_tokens, temperature, messages))
        if self._raise is not None:
            raise self._raise
        return self._resp


class TestLlmClient:
    def test_protocol_conformance(self, base_config):
        client = AnthropicLlmClient(
            config=base_config.briefing.llm, client=_FakeAnthropic(resp=_Resp(["x"]))
        )
        assert isinstance(client, LlmClient)

    def test_generate_extracts_and_joins_text(self, base_config):
        fake = _FakeAnthropic(resp=_Resp(["Hello ", "world"]))
        client = AnthropicLlmClient(config=base_config.briefing.llm, client=fake)
        assert client.generate(prompt="p") == "Hello world"

    def test_generate_passes_config_params(self, base_config):
        fake = _FakeAnthropic(resp=_Resp(["x"]))
        client = AnthropicLlmClient(config=base_config.briefing.llm, client=fake)
        client.generate(prompt="p")
        model, max_tokens, temperature, messages = fake.calls[0]
        assert model == base_config.briefing.llm.model
        assert max_tokens == base_config.briefing.llm.max_tokens
        assert temperature == base_config.briefing.llm.temperature
        assert messages == [{"role": "user", "content": "p"}]

    def test_empty_response_raises(self, base_config):
        client = AnthropicLlmClient(
            config=base_config.briefing.llm, client=_FakeAnthropic(resp=_Resp([]))
        )
        with pytest.raises(BriefingLlmError):
            client.generate(prompt="p")

    def test_sdk_error_wrapped_and_redacted(self, base_config):
        fake = _FakeAnthropic(raise_exc=Exception("boom apiKey=SECRETKEY123"))
        client = AnthropicLlmClient(config=base_config.briefing.llm, client=fake)
        with pytest.raises(BriefingLlmError) as exc_info:
            client.generate(prompt="p")
        msg = str(exc_info.value)
        assert "SECRETKEY123" not in msg
        assert "***" in msg


# ---------------------------------------------------------------------------
# Email client
# ---------------------------------------------------------------------------
def _set_smtp_env(monkeypatch, *, recipients="a@example.com, b@example.com"):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("BRIEFING_RECIPIENTS", recipients)


class _Recorder:
    def __init__(self, *, raise_exc=None):
        self._raise = raise_exc
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self._raise is not None:
            raise self._raise


class TestEmailClient:
    def test_protocol_conformance(self, base_config, monkeypatch):
        _set_smtp_env(monkeypatch)
        client = SmtpEmailClient(config=base_config.briefing.email, sender=_Recorder())
        assert isinstance(client, EmailClient)

    def test_recipients_parsed_from_env(self, base_config, monkeypatch):
        _set_smtp_env(monkeypatch, recipients="a@example.com, b@example.com ,")
        client = SmtpEmailClient(config=base_config.briefing.email, sender=_Recorder())
        assert client.recipients == ("a@example.com", "b@example.com")

    def test_send_invokes_transport_with_resolved_fields(self, base_config, monkeypatch):
        _set_smtp_env(monkeypatch)
        rec = _Recorder()
        client = SmtpEmailClient(config=base_config.briefing.email, sender=rec)
        client.send(subject="S", body="B")
        assert rec.kwargs["host"] == "smtp.example.com"
        assert rec.kwargs["subject"] == "S"
        assert rec.kwargs["body"] == "B"
        assert rec.kwargs["recipients"] == ["a@example.com", "b@example.com"]

    def test_empty_recipients_raises(self, base_config, monkeypatch):
        _set_smtp_env(monkeypatch, recipients=" , ")
        with pytest.raises(BriefingEmailError):
            SmtpEmailClient(config=base_config.briefing.email, sender=_Recorder())

    def test_send_error_wrapped_and_redacted(self, base_config, monkeypatch):
        _set_smtp_env(monkeypatch)
        rec = _Recorder(raise_exc=Exception("auth failed password=PWSECRET999"))
        client = SmtpEmailClient(config=base_config.briefing.email, sender=rec)
        with pytest.raises(BriefingEmailError) as exc_info:
            client.send(subject="S", body="B")
        msg = str(exc_info.value)
        assert "PWSECRET999" not in msg
        assert "***" in msg
