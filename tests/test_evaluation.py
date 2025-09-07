from pathlib import Path

import pytest

from local_llm_harness.config import EvaluationSettings, ModelProfile
from local_llm_harness.contracts import EvaluationResult
from local_llm_harness.evaluation import (
    BenchmarkManifest,
    EvaluationBudget,
    EvaluationError,
    EvaluationHarness,
    EvaluationReport,
    EvaluationStore,
)


def budget() -> EvaluationBudget:
    return EvaluationBudget(
        model="fake/model",
        model_timeout_seconds=30,
        max_tokens_per_call=1000,
        max_model_calls=10,
        max_total_tokens=10_000,
        run_timeout_seconds=300,
        command_timeout_seconds=60,
        max_commands=5,
        max_agent_steps=8,
        cpu_limit=1,
        memory_mb=512,
        pids_limit=32,
        disk_mb=128,
        tmpfs_mb=32,
        network_disabled=True,
    )


def manifest(*instances: str) -> BenchmarkManifest:
    return BenchmarkManifest(
        name="fixture",
        dataset_name="fixture/dataset",
        split="test",
        swebench_version="4.0.3",
        instances=instances,
    )


class FixtureRunner:
    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.calls = []
        self.interrupt_on_call = interrupt_on_call

    async def run_instance(self, instance_id, mode, model_profile, resource_budget, evaluation_dir):
        del evaluation_dir
        self.calls.append((instance_id, mode, resource_budget.digest))
        if self.interrupt_on_call == len(self.calls):
            raise KeyboardInterrupt
        return EvaluationResult(
            instance_id=instance_id,
            model_profile=model_profile,
            mode=mode,
            resolved=mode == "full",
            duration_seconds=2,
            prompt_tokens=10,
            completion_tokens=5,
            model_calls=2,
            research_requests=1 if mode == "full" else 0,
        )


@pytest.mark.asyncio
async def test_evaluation_aggregates_metrics_and_writes_reports(tmp_path: Path) -> None:
    runner = FixtureRunner()
    store = EvaluationStore(tmp_path)
    run = await EvaluationHarness(store, runner).run(
        manifest("owner__repo-1", "owner__repo-2"), "local", budget()
    )

    report = EvaluationReport.from_run(run)

    assert run.complete
    assert len(run.records) == 4
    assert report.metrics[0].resolution_rate == 0
    assert report.metrics[1].resolution_rate == 1
    assert report.metrics[1].research_requests == 2
    directory = store.directory(run.evaluation_id)
    assert (directory / "results.json").is_file()
    markdown = (directory / "comparison.md").read_text(encoding="utf-8")
    assert "baseline" in markdown
    assert "full" in markdown
    assert budget().digest in markdown


@pytest.mark.asyncio
async def test_interrupted_evaluation_resumes_only_missing_work(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    first = FixtureRunner(interrupt_on_call=2)
    harness = EvaluationHarness(store, first)
    selected_manifest = manifest("owner__repo-1", "owner__repo-2")

    with pytest.raises(KeyboardInterrupt):
        await harness.run(selected_manifest, "local", budget())

    evaluation_dirs = list((tmp_path / "evaluations").iterdir())
    persisted = store.load(evaluation_dirs[0].name)
    assert len(persisted.records) == 1

    resumed_runner = FixtureRunner()
    resumed = await EvaluationHarness(store, resumed_runner).run(
        selected_manifest,
        "local",
        budget(),
        evaluation_id=persisted.evaluation_id,
    )

    assert resumed.complete
    assert len(resumed_runner.calls) == 3
    assert resumed_runner.calls[0][:2] == ("owner__repo-1", "full")


@pytest.mark.asyncio
async def test_resume_rejects_changed_resource_budget(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    run = store.create(manifest("owner__repo-1"), "local", ("baseline",), budget())
    changed = budget().model_copy(update={"max_total_tokens": 20_000})

    with pytest.raises(EvaluationError, match="resource budget"):
        await EvaluationHarness(store, FixtureRunner()).run(
            run.manifest,
            "local",
            changed,
            modes=("baseline",),
            evaluation_id=run.evaluation_id,
        )


def test_supplied_manifest_matches_the_ten_task_experiment() -> None:
    supplied = BenchmarkManifest.load(Path("benchmarks/swebench_lite_10.yaml"))

    assert supplied.swebench_version == "4.0.3"
    assert len(supplied.instances) == 10
    assert supplied.instances[0] == "pvlib__pvlib-python-1707"
    assert supplied.instances[-1] == "matplotlib__matplotlib-22711"


def test_budget_captures_model_and_sandbox_limits() -> None:
    from local_llm_harness.config import DockerSettings

    captured = EvaluationBudget.from_settings(
        ModelProfile(model="fake/model", max_tokens=1234),
        DockerSettings(cpu_limit=1.5, memory_mb=768),
        EvaluationSettings(max_model_calls=7, max_total_tokens=9000),
    )

    assert captured.max_tokens_per_call == 1234
    assert captured.max_model_calls == 7
    assert captured.max_total_tokens == 9000
    assert captured.cpu_limit == 1.5
    assert captured.memory_mb == 768
