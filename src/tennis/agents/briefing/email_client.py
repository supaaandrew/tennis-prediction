"""Email client boundary for the Briefing Agent (§N2).

`EmailClient` is the `@runtime_checkable` Protocol the agent depends on — it
never imports `smtplib` directly. The real `SmtpEmailClient` owns delivery: it
resolves recipients + credentials from env via `read_required_env` at
construction and sends over SMTP (+STARTTLS per config). Any transport failure is
wrapped in a typed `BriefingEmailError` with a `redact_text`'d cause (§L10). Unit
tests inject a mock; the suite never sends real email.
"""

from __future__ import annotations

import smtplib
from collections.abc import Callable, Sequence
from email.message import EmailMessage
from typing import Protocol, runtime_checkable

from tennis.core.config import EmailConfig, read_required_env
from tennis.core.errors import BriefingEmailError
from tennis.core.logging import get_logger, redact_text

_logger = get_logger("tennis.agents.briefing.email_client")

# An injectable transport: send a fully-formed message. Defaults to real SMTP;
# tests pass a fake that records or raises (no network).
SmtpSender = Callable[..., None]


@runtime_checkable
class EmailClient(Protocol):
    """Minimal delivery surface. The client owns where mail goes (recipients)."""

    def send(self, *, subject: str, body: str) -> None: ...


class SmtpEmailClient:
    """SMTP-backed `EmailClient`.

    Credentials (`host_env`/`user_env`/`password_env`) and recipients
    (`recipients_env`, comma-separated) come from env at construction; `port`,
    `from`, and `tls` come from `EmailConfig`. The SMTP transport is injectable
    (`sender=`) for testing. `send` wraps any failure in `BriefingEmailError`
    with a redacted cause.
    """

    def __init__(
        self, *, config: EmailConfig, sender: SmtpSender | None = None
    ) -> None:
        self._config = config
        self._host = read_required_env(config.smtp.host_env)
        self._user = read_required_env(config.smtp.user_env)
        self._password = read_required_env(config.smtp.password_env)
        self._recipients = _parse_recipients(read_required_env(config.recipients_env))
        if not self._recipients:
            raise BriefingEmailError(
                f"{config.recipients_env} resolved to zero recipients"
            )
        self._sender: SmtpSender = sender or _default_smtp_send

    @property
    def recipients(self) -> tuple[str, ...]:
        return tuple(self._recipients)

    def send(self, *, subject: str, body: str) -> None:
        try:
            self._sender(
                host=self._host,
                port=self._config.smtp.port,
                user=self._user,
                password=self._password,
                tls=self._config.smtp.tls,
                sender=self._config.smtp.from_,
                recipients=self._recipients,
                subject=subject,
                body=body,
            )
        except BriefingEmailError:
            raise
        except Exception as exc:  # connect/auth/send — wrap + redact (§L10)
            raise BriefingEmailError(redact_text(str(exc))) from exc


def _parse_recipients(raw: str) -> list[str]:
    return [r.strip() for r in raw.split(",") if r.strip()]


def _default_smtp_send(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    tls: bool,
    sender: str,
    recipients: Sequence[str],
    subject: str,
    body: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)
    with smtplib.SMTP(host, port) as smtp:
        if tls:
            smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
