"""CLI entrypoint (§S2/§S3) — `tennis train` and `tennis run`.

Two commands, each building its chain via the composition root and running it
once under one `run_id`:

  - `train` — Data → Research(training) → Modeling(training). Ingests, trains,
    and registers/activates a model. Run on demand (or a slower cron); it is the
    prerequisite for the daily chain (bootstrap, §S6).
  - `run` — Data → Research(prediction) → Modeling(prediction) → Briefing →
    Monitor. The 06:30 UTC daily prediction cron.

Exit code (§S8): 0 when every agent that ran ended `succeeded`/`partial`; 1 when
any agent that ran ended `failed`. A `None` result (another run already holds the
singleton lock, §L7) is a clean no-op → 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from tennis.composition import build_daily_chain, build_training_chain
from tennis.core.config import load_config

if TYPE_CHECKING:
    from tennis.core.lineage import RunStatus

_DEFAULT_CONFIG = "config/config.yaml"

app = typer.Typer(add_completion=False, help="Tennis prediction bot pipelines.")


def _exit_code(status: RunStatus | None) -> int:
    """§S8: the aggregate run status is `failed` iff some agent that ran failed;
    `partial` (including Monitor's "not enough data yet") and `succeeded` → 0.
    `None` means another run holds the lock (benign no-op) → 0."""
    return 1 if status == "failed" else 0


@app.command()
def train(
    config: str = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Train + register a model (Data -> Research(training) -> Modeling(training))."""
    cfg = load_config(config)
    status = build_training_chain(cfg).run_once()
    typer.echo(f"train: {status}")
    raise typer.Exit(code=_exit_code(status))


@app.command()
def run(
    config: str = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Daily prediction chain (Data -> Research -> Modeling -> Briefing -> Monitor)."""
    cfg = load_config(config)
    status = build_daily_chain(cfg).run_once()
    typer.echo(f"run: {status}")
    raise typer.Exit(code=_exit_code(status))


if __name__ == "__main__":  # pragma: no cover
    app()
