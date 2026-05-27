"""`python -m tennis …` entrypoint — delegates to the typer CLI app."""

from __future__ import annotations

from tennis.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
