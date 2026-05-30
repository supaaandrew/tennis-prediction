"""Unit tests for the scripts/sync_sackmann_mirror.py ops utility.

The script is not an importable package, so it is loaded by path. `sync_mirror`
takes injectable `git` / `touch` callables, so these tests exercise the
clone-vs-pull-vs-pin branching and the mtime-refresh contract without touching
git or the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tennis.core.config import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "sync_sackmann_mirror.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("sync_sackmann_mirror", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sync_mod = _load_module()


class _FakeGit:
    def __init__(self, head: str = "abc1234") -> None:
        self.calls: list[tuple[list[str], object]] = []
        self._head = head

    def __call__(self, args, *, cwd=None):  # type: ignore[no-untyped-def]
        self.calls.append((list(args), cwd))
        if args and args[0] == "rev-parse":
            return self._head
        return ""

    @property
    def commands(self) -> list[list[str]]:
        return [c[0] for c in self.calls]


class TestSyncMirror:
    def test_clones_when_no_git_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "mirror"
        git = _FakeGit()
        touched: list[Path] = []

        sync_mod.sync_mirror(
            repo="Jeff/tennis_atp",
            branch="master",
            local_mirror_dir=target,
            pin_commit_sha=None,
            git=git,
            touch=touched.append,
        )

        assert git.commands[0][:2] == ["clone", "--branch"]
        assert "https://github.com/Jeff/tennis_atp.git" in git.commands[0]
        assert not any(c[0] == "pull" for c in git.commands)
        assert touched == [target]

    def test_pulls_ff_only_when_git_dir_exists(self, tmp_path: Path) -> None:
        target = tmp_path / "mirror"
        (target / ".git").mkdir(parents=True)
        git = _FakeGit()
        touched: list[Path] = []

        sync_mod.sync_mirror(
            repo="Jeff/tennis_atp",
            branch="master",
            local_mirror_dir=target,
            pin_commit_sha=None,
            git=git,
            touch=touched.append,
        )

        assert ["fetch", "origin"] in git.commands
        assert ["pull", "--ff-only", "origin", "master"] in git.commands
        assert not any(c[0] == "clone" for c in git.commands)
        assert touched == [target]

    def test_pinned_sha_fetches_and_checks_out_without_pull(self, tmp_path: Path) -> None:
        target = tmp_path / "mirror"
        (target / ".git").mkdir(parents=True)
        git = _FakeGit()

        sync_mod.sync_mirror(
            repo="Jeff/tennis_atp",
            branch="master",
            local_mirror_dir=target,
            pin_commit_sha="deadbee",
            git=git,
            touch=lambda _p: None,
        )

        assert ["fetch", "origin"] in git.commands
        assert not any(c[0] == "pull" for c in git.commands)
        assert ["checkout", "deadbee"] in git.commands

    def test_config_provides_branch_field_the_script_depends_on(self) -> None:
        # main() reads config.sources.sackmann.branch; lock that the field exists
        # and defaults to "master" so the sync script never breaks on it.
        cfg = load_config(_REPO_ROOT / "config" / "config.yaml")
        assert cfg.sources.sackmann.branch == "master"
        assert isinstance(cfg.sources.sackmann.repo, str) and cfg.sources.sackmann.repo

    def test_missing_git_binary_becomes_git_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # git not on PATH -> FileNotFoundError -> converted to GitError so main()
        # reports it via the single structured nonzero-exit path.
        def _raise(*_a: object, **_k: object) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr(sync_mod.subprocess, "run", _raise)
        with pytest.raises(sync_mod.GitError):
            sync_mod._run_git(["rev-parse", "HEAD"])

    def test_git_timeout_becomes_git_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A hung/slow git -> TimeoutExpired -> GitError (never an indefinite block
        # in a scheduled run).
        def _raise(*_a: object, **_k: object) -> None:
            raise sync_mod.subprocess.TimeoutExpired(cmd="git", timeout=sync_mod._GIT_TIMEOUT_S)

        monkeypatch.setattr(sync_mod.subprocess, "run", _raise)
        with pytest.raises(sync_mod.GitError):
            sync_mod._run_git(["pull", "--ff-only", "origin", "master"])

    def test_git_error_propagates_and_skips_touch(self, tmp_path: Path) -> None:
        target = tmp_path / "mirror"
        (target / ".git").mkdir(parents=True)
        touched: list[Path] = []

        def _boom(args, *, cwd=None):  # type: ignore[no-untyped-def]
            raise sync_mod.GitError("git exploded")

        with pytest.raises(sync_mod.GitError):
            sync_mod.sync_mirror(
                repo="Jeff/tennis_atp",
                branch="master",
                local_mirror_dir=target,
                pin_commit_sha=None,
                git=_boom,
                touch=touched.append,
            )

        assert touched == []
