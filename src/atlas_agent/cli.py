"""OMOP Atlas Agent command-line entry point.

Wires the four operational entry points behind a single Typer application so
users invoke them as::

    atlas_agent query "<clinical description>" [--output FILE]
    atlas_agent challenge <CHALLENGE_ID>
    atlas_agent challenges [--output DIR]
    atlas_agent vignettes <FILES_OR_DIRS...> [--output DIR]

instead of the older ``uv run python -m atlas_agent.main ...`` /
``uv run python run_*.py ...`` forms.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from . import main as main_module
from ._logging import setup_logging
from ._runners import challenges as challenges_runner
from ._runners import vignettes as vignettes_runner

app = typer.Typer(
    name="atlas_agent",
    help="OMOP Atlas Agent — natural-language phenotypes to ATLAS concept sets.",
    no_args_is_help=True,
    add_completion=False,
)


def _configure(log_level: Optional[str]) -> None:
    """Apply the optional --log-level override and configure logging once."""
    setup_logging(level=log_level.upper() if log_level else None)


@app.command()
def query(
    description: str = typer.Argument(..., help="Clinical phenotype description (free text)."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the ATLAS JSON to this path (default: stdout summary only).",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Override LOG_LEVEL (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
) -> None:
    """Run the orchestrator on a single inline clinical description."""
    _configure(log_level)
    raise typer.Exit(code=main_module.query(description, export_path=output))


@app.command()
def challenge(
    challenge_id: str = typer.Argument(..., help="Challenge ID (e.g. C01, C02, ..., C07)."),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Override LOG_LEVEL (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
) -> None:
    """Run one Mind Meets Machines challenge by ID (C01–C07)."""
    _configure(log_level)
    raise typer.Exit(code=challenges_runner.run_single_challenge(challenge_id))


@app.command()
def challenges(
    output: Path = typer.Option(
        Path("output/challenges"),
        "--output",
        "-o",
        help="Output directory for per-challenge artefacts.",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Override LOG_LEVEL (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
) -> None:
    """Run every Mind Meets Machines challenge end-to-end."""
    _configure(log_level)
    raise typer.Exit(code=challenges_runner.run_all_challenges(output_dir=output))


@app.command()
def vignettes(
    inputs: List[str] = typer.Argument(..., help="One or more .md files or directories."),
    output: Path = typer.Option(
        Path("output/vignettes"),
        "--output",
        "-o",
        help="Output directory for per-vignette artefacts.",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Override LOG_LEVEL (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
) -> None:
    """Process custom patient vignettes from local .md files."""
    _configure(log_level)
    raise typer.Exit(code=vignettes_runner.run_vignettes(inputs, output_dir=output))


if __name__ == "__main__":
    app()
