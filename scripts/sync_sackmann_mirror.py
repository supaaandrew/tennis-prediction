"""Sync the local Sackmann CSV mirror (DEV/OPS utility, not a runtime import).

`FilesystemMirrorReader` (adapters/sackmann/adapter.py) reads the Sackmann ATP
CSVs from `config.sources.sackmann.local_mirror_dir` — it expects a git clone of
the repo on disk and does NOT clone it itself. This script is the clone/pull step
the daily wrappers (ops/) run before `python -m tennis train` / `run`.

Critically it refreshes the mirror directory's mtime after a pull. The §C5
staleness pre-flight halts the pipeline (`SackmannStalenessError`) when
`dir_mtime()` is older than `max_staleness_days`, and a `git pull` that changes
only files INSIDE the dir does not bump the top-level directory mtime — so we
`os.utime(dir)` explicitly. (This is an OS mtime touch, not a `datetime.now()`
call; the no-bare-`datetime.now()` rule is about app logic, not filesystem ops.)

Exit code: 0 on success, 1 on any git failure (so the ops wrapper aborts before
attempting a run against a stale/absent mirror).

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/sync_sackmann_mirror.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Protocol, Sequence

import structlog

from tennis.core.config import load_config

logger = structlog.get_logger(__name__)

_DEFAULT_CONFIG = "config/config.yaml"
_GITHUB_BASE = "https://github.com"
# Generous enough for the large initial clone of the full Sackmann repo, but
# still bounds a hung/stalled git so a scheduled run cannot block forever.
_GIT_TIMEOUT_S = 900


class GitRunner(Protocol):
    def __call__(self, args: Sequence[str], *, cwd: Path | None = None) -> str: ...


class GitError(RuntimeError):
    """A git subprocess exited non-zero."""


def _run_git(args: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run `git <args>` and return stdout; raise GitError on ANY failure.

    A missing git binary (`FileNotFoundError`) and a hung/slow git
    (`TimeoutExpired`) are both converted to `GitError` so every failure mode
    follows the one structured nonzero-exit path in `main()` — never an uncaught
    traceback or an indefinite block in a scheduled run.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def _touch_dir(path: Path) -> None:
    os.utime(path, None)


def sync_mirror(
    *,
    repo: str,
    branch: str,
    local_mirror_dir: str | Path,
    pin_commit_sha: str | None = None,
    git: GitRunner = _run_git,
    touch: Callable[[Path], None] = _touch_dir,
) -> Path:
    """Clone or fast-forward the Sackmann mirror, then refresh its dir mtime.

    - No `.git` in the target dir -> `git clone --branch <branch> <repo> <dir>`.
    - Otherwise -> `git fetch` + `git pull --ff-only` on the configured branch.
    - `pin_commit_sha` set -> `git checkout <sha>` (lock to a known revision; the
      pull is skipped because a pinned mirror is intentionally frozen).
    - Always `os.utime(<dir>)` afterwards so §C5 staleness sees a fresh mirror.

    Returns the resolved mirror Path. Raises GitError on any git failure.
    """
    target = Path(local_mirror_dir)
    url = f"{_GITHUB_BASE}/{repo}.git"
    is_clone = not (target / ".git").is_dir()

    if is_clone:
        logger.info("sackmann_mirror_clone", repo=repo, branch=branch, dir=str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        git(["clone", "--branch", branch, url, str(target)])
    elif pin_commit_sha:
        logger.info("sackmann_mirror_pinned_fetch", sha=pin_commit_sha, dir=str(target))
        git(["fetch", "origin"], cwd=target)
    else:
        logger.info("sackmann_mirror_pull", repo=repo, branch=branch, dir=str(target))
        git(["fetch", "origin"], cwd=target)
        git(["pull", "--ff-only", "origin", branch], cwd=target)

    if pin_commit_sha:
        logger.info("sackmann_mirror_checkout", sha=pin_commit_sha, dir=str(target))
        git(["checkout", pin_commit_sha], cwd=target)

    touch(target)
    head = git(["rev-parse", "--short", "HEAD"], cwd=target)
    logger.info("sackmann_mirror_synced", dir=str(target), head=head)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync the local Sackmann CSV mirror.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to config.yaml")
    ns = parser.parse_args(argv)

    cfg = load_config(ns.config)
    sackmann = cfg.sources.sackmann
    try:
        target = sync_mirror(
            repo=sackmann.repo,
            branch=sackmann.branch,
            local_mirror_dir=sackmann.local_mirror_dir,
            pin_commit_sha=sackmann.pin_commit_sha,
        )
    except GitError as exc:
        logger.error("sackmann_mirror_sync_failed", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Sackmann mirror synced: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
