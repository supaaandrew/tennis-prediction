"""CLI tests (§S2/§S3/§S8).

The composition root and `run_once` are mocked — no chain is built or run — so
these assert the command→builder mapping and the §S8 exit-code rule only.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from tennis import cli

runner = CliRunner()


class _FakePipeline:
    def __init__(self, status: Any) -> None:
        self._status = status
        self.ran = False

    def run_once(self) -> Any:
        self.ran = True
        return self._status


def _patch(monkeypatch: pytest.MonkeyPatch, *, which: str, status: Any):
    """Patch load_config + the named builder to return a fake pipeline; make the
    OTHER builder explode so we prove the command picked the right chain."""
    captured: dict[str, Any] = {}
    fake = _FakePipeline(status)
    monkeypatch.setattr(cli, "load_config", lambda path: f"CFG::{path}")

    def _build(cfg: Any) -> _FakePipeline:
        captured["cfg"] = cfg
        return fake

    other = "build_daily_chain" if which == "build_training_chain" else "build_training_chain"
    monkeypatch.setattr(cli, which, _build)

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(f"{other} must not be called")

    monkeypatch.setattr(cli, other, _boom)
    return captured, fake


class TestExitCodeRule:
    @pytest.mark.parametrize(
        "status,expected",
        [("succeeded", 0), ("partial", 0), ("failed", 1), (None, 0)],
    )
    def test_exit_code(self, status: Any, expected: int) -> None:
        assert cli._exit_code(status) == expected


class TestTrainCommand:
    def test_train_builds_training_chain_and_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured, fake = _patch(monkeypatch, which="build_training_chain", status="succeeded")
        result = runner.invoke(cli.app, ["train", "--config", "x.yaml"])
        assert result.exit_code == 0
        assert fake.ran
        assert captured["cfg"] == "CFG::x.yaml"

    def test_train_failed_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, which="build_training_chain", status="failed")
        result = runner.invoke(cli.app, ["train"])
        assert result.exit_code == 1


class TestRunCommand:
    def test_run_builds_daily_chain_and_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured, fake = _patch(monkeypatch, which="build_daily_chain", status="partial")
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 0  # partial never forces nonzero (§S8)
        assert fake.ran

    def test_run_failed_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, which="build_daily_chain", status="failed")
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 1

    def test_run_lock_held_none_is_clean_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, which="build_daily_chain", status=None)
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 0
