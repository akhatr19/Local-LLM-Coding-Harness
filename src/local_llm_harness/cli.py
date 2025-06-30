"""Command-line interface for the coding harness."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from local_llm_harness import __version__
from local_llm_harness.config import load_settings

app = typer.Typer(
    name="harness",
    help="A staged coding harness for small language models.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed harness version."""

    typer.echo(__version__)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


@app.command()
def doctor(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            help="Optional YAML configuration file.",
        ),
    ] = None,
    show_config: Annotated[
        bool,
        typer.Option(
            "--show-config",
            help="Print the effective configuration with secrets redacted.",
        ),
    ] = False,
) -> None:
    """Validate the checkpoint-1 runtime and configuration."""

    try:
        settings = load_settings(config)
    except (ValueError, ValidationError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    artifact_parent = _nearest_existing_parent(settings.artifacts.root)
    checks = [
        ("Python", sys.version_info >= (3, 11), platform.python_version()),
        ("Configuration", True, "valid"),
        (
            "Default model profile",
            settings.litellm.default_profile in settings.litellm.profiles,
            settings.litellm.default_profile,
        ),
        (
            "Artifact storage",
            os.access(artifact_parent, os.W_OK),
            str(artifact_parent),
        ),
    ]

    table = Table(title="Harness doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    for name, passed, details in checks:
        table.add_row(name, "PASS" if passed else "FAIL", details)
    console.print(table)

    if show_config:
        console.print_json(json.dumps(settings.redacted_dict()))
    if not all(passed for _, passed, _ in checks):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
