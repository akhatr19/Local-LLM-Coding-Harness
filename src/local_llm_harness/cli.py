"""Command-line interface for the coding harness."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from local_llm_harness import __version__
from local_llm_harness.coding import CodingStatus
from local_llm_harness.config import load_settings
from local_llm_harness.evaluation import (
    BenchmarkManifest,
    EvaluationBudget,
    EvaluationError,
    EvaluationHarness,
    EvaluationReport,
    EvaluationStore,
)
from local_llm_harness.indexing import RepositoryIndexer, SentenceTransformerEmbedder
from local_llm_harness.investigation import InvestigationError
from local_llm_harness.model_gateway import LiteLLMGateway, ModelGatewayError
from local_llm_harness.pipeline import run_full_pipeline
from local_llm_harness.planning import PlanningError
from local_llm_harness.repository import RepositoryError, RepositoryInspector
from local_llm_harness.sandbox import SandboxError
from local_llm_harness.storage import RunNotFoundError, RunStore, RunStoreError
from local_llm_harness.swebench import SWEbenchComparisonRunner

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
    check_model: Annotated[
        bool,
        typer.Option(
            "--check-model",
            help="Make one minimal live call to the configured default model.",
        ),
    ] = False,
) -> None:
    """Validate the local runtime and optionally contact the default model."""

    try:
        settings = load_settings(config)
    except (ValueError, ValidationError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    artifact_parent = _nearest_existing_parent(settings.artifacts.root)
    checks: list[tuple[str, str, str]] = [
        ("Python", "PASS" if sys.version_info >= (3, 11) else "FAIL", platform.python_version()),
        ("Configuration", "PASS", "valid"),
        (
            "Default model profile",
            "PASS" if settings.litellm.default_profile in settings.litellm.profiles else "FAIL",
            settings.litellm.default_profile,
        ),
        (
            "LiteLLM package",
            "PASS" if importlib.util.find_spec("litellm") is not None else "FAIL",
            "installed" if importlib.util.find_spec("litellm") is not None else "missing",
        ),
        (
            "Git",
            "PASS" if shutil.which("git") is not None else "FAIL",
            shutil.which("git") or "missing",
        ),
        (
            "ripgrep",
            "PASS" if shutil.which("rg") is not None else "FAIL",
            shutil.which("rg") or "missing",
        ),
        (
            "ChromaDB package",
            "PASS" if importlib.util.find_spec("chromadb") is not None else "FAIL",
            "installed" if importlib.util.find_spec("chromadb") is not None else "missing",
        ),
        (
            "Sentence Transformers package",
            "PASS" if importlib.util.find_spec("sentence_transformers") is not None else "FAIL",
            "installed"
            if importlib.util.find_spec("sentence_transformers") is not None
            else "missing",
        ),
        (
            "Artifact storage",
            "PASS" if os.access(artifact_parent, os.W_OK) else "FAIL",
            str(artifact_parent),
        ),
    ]

    if check_model:
        try:
            model = asyncio.run(LiteLLMGateway(settings.litellm).check_connection())
            checks.append(("Model connectivity", "PASS", model))
        except ModelGatewayError as exc:
            checks.append(("Model connectivity", "FAIL", str(exc)))
    else:
        checks.append(("Model connectivity", "SKIP", "use --check-model to enable"))

    table = Table(title="Harness doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    for name, status, details in checks:
        table.add_row(name, status, details)
    console.print(table)

    if show_config:
        console.print_json(json.dumps(settings.redacted_dict()))
    if any(status == "FAIL" for _, status, _ in checks):
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect_run(
    run_id: Annotated[UUID, typer.Argument(help="Run UUID to inspect.")],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            help="Optional YAML configuration file.",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete run state as JSON."),
    ] = False,
) -> None:
    """Show persisted state and artifacts for a run."""

    try:
        settings = load_settings(config)
        store = RunStore(settings.artifacts.root)
        state = store.get_run(run_id)
    except (ValueError, ValidationError, RunNotFoundError) as exc:
        console.print(f"[red]Unable to inspect run:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(state.model_dump_json())
        return

    summary = Table(title=f"Run {state.run_id}")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Task", state.task_id)
    summary.add_row("Status", state.status.value)
    summary.add_row("Current stage", state.current_stage.value)
    summary.add_row("Created", state.created_at.isoformat())
    summary.add_row("Updated", state.updated_at.isoformat())
    console.print(summary)

    if state.stages:
        stages = Table(title="Stage attempts")
        stages.add_column("Stage")
        stages.add_column("Status")
        stages.add_column("Started")
        stages.add_column("Finished")
        for attempt in state.stages:
            stages.add_row(
                attempt.stage.value,
                attempt.status.value,
                attempt.started_at.isoformat(),
                attempt.finished_at.isoformat() if attempt.finished_at else "-",
            )
        console.print(stages)

    artifacts = store.list_artifacts(run_id)
    console.print("Artifacts:")
    if artifacts:
        for artifact in artifacts:
            console.print(f"- {artifact.relative_to(store.root)}")
    else:
        console.print("- none")


@app.command("index")
def index_repository(
    repository: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Repository root to index.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            help="Optional YAML configuration file.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild even when the fingerprint is unchanged."),
    ] = False,
) -> None:
    """Build or reuse the persistent hybrid retrieval index."""

    try:
        settings = load_settings(config)
        inspector = RepositoryInspector(repository)
        embedder = SentenceTransformerEmbedder(settings.retrieval.embedding_model)
        indexer = RepositoryIndexer(
            inspector,
            settings.retrieval,
            settings.artifacts.root / "indexes",
            embedder,
        )
        outcome = indexer.index(force=force)
    except (ValueError, ValidationError, RepositoryError, RuntimeError) as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    action = "Reused" if outcome.reused else "Built"
    console.print(f"[green]{action} repository index[/green]")
    console.print(f"Repository: {outcome.manifest.repository}")
    console.print(f"Commit: {outcome.manifest.commit or 'uncommitted'}")
    console.print(f"Chunks: {outcome.manifest.chunk_count}")
    console.print(f"Embedding model: {outcome.manifest.embedding_model}")


@app.command("run")
def run_pipeline(
    repository: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Clean Git repository to investigate and modify in a disposable sandbox.",
        ),
    ],
    issue_file: Annotated[
        Path,
        typer.Option(
            "--issue-file",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="UTF-8 file containing the coding issue.",
        ),
    ],
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Configured LiteLLM model profile."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            help="Optional YAML configuration file.",
        ),
    ] = None,
    run_id: Annotated[
        UUID | None,
        typer.Option("--run-id", help="Resume an existing run."),
    ] = None,
) -> None:
    """Run investigation, planning, research, and sandboxed implementation."""

    try:
        if issue_file.stat().st_size > 1_000_000:
            raise ValueError("issue file exceeds the 1 MB limit")
        issue = issue_file.read_text(encoding="utf-8").strip()
        if not issue:
            raise ValueError("issue file cannot be empty")
        settings = load_settings(config)
        selected_profile = profile or settings.litellm.default_profile
        if selected_profile not in settings.litellm.profiles:
            raise ValueError(f"unknown model profile: {selected_profile}")
        result = asyncio.run(
            run_full_pipeline(
                repository=repository,
                issue=issue,
                profile=selected_profile,
                settings=settings,
                run_id=run_id,
            )
        )
    except (
        UnicodeDecodeError,
        ValueError,
        ValidationError,
        ModelGatewayError,
        RepositoryError,
        RunStoreError,
        InvestigationError,
        PlanningError,
        SandboxError,
        RuntimeError,
    ) as exc:
        console.print(f"[red]Run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Run: {result.run_id}")
    console.print(f"Implementation status: {result.result.status.value}")
    console.print(f"Changed files: {len(result.result.changed_files)}")
    console.print(f"Patch artifact: runs/{result.run_id}/implementation/changes.patch")
    if result.result.status is CodingStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("eval")
def evaluate(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Benchmark manifest; defaults to the supplied ten-instance experiment.",
        ),
    ] = Path("benchmarks/swebench_lite_10.yaml"),
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Configured LiteLLM model profile."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Comparison mode: baseline, full, or both."),
    ] = "both",
    config: Annotated[
        Path | None,
        typer.Option("--config", dir_okay=False, help="Optional YAML configuration file."),
    ] = None,
    resume: Annotated[
        UUID | None,
        typer.Option("--resume", help="Resume a persisted evaluation UUID."),
    ] = None,
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Run only the first manifest instance in both modes."),
    ] = False,
) -> None:
    """Compare direct baseline and full pipeline on SWE-bench 4.0.3."""

    try:
        settings = load_settings(config)
        selected_profile = profile or settings.litellm.default_profile
        if selected_profile not in settings.litellm.profiles:
            raise ValueError(f"unknown model profile: {selected_profile}")
        if mode not in {"baseline", "full", "both"}:
            raise ValueError("mode must be baseline, full, or both")
        modes = ("baseline", "full") if mode == "both" else (mode,)
        benchmark = BenchmarkManifest.load(manifest)
        if benchmark.swebench_version != settings.evaluation.swebench_version:
            raise ValueError("manifest SWE-bench version differs from configuration")
        if smoke:
            benchmark = benchmark.model_copy(
                update={"name": f"{benchmark.name}-smoke", "instances": benchmark.instances[:1]}
            )
        budget = EvaluationBudget.from_settings(
            settings.litellm.profiles[selected_profile],
            settings.docker,
            settings.evaluation,
        )
        store = EvaluationStore(settings.artifacts.root)
        run = asyncio.run(
            EvaluationHarness(store, SWEbenchComparisonRunner(settings)).run(
                benchmark,
                selected_profile,
                budget,
                modes=modes,
                evaluation_id=resume,
            )
        )
    except (ValueError, ValidationError, EvaluationError) as exc:
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    report = EvaluationReport.from_run(run)
    console.print(f"Evaluation: {run.evaluation_id}")
    for metrics in report.metrics:
        console.print(
            f"{metrics.mode}: {metrics.resolved}/{metrics.attempts} ({metrics.resolution_rate:.1%})"
        )
    console.print(f"Reports: evaluations/{run.evaluation_id}/results.json and comparison.md")


if __name__ == "__main__":
    app()
