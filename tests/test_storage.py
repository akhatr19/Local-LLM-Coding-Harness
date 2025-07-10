from pathlib import Path

import pytest

from local_llm_harness.contracts import RunStage, RunStatus, TaskSpec
from local_llm_harness.storage import (
    InvalidTransitionError,
    RunStore,
    UnsafeArtifactPathError,
)


def task() -> TaskSpec:
    return TaskSpec(
        task_id="issue-1",
        repository=Path("/tmp/example"),
        problem_statement="Fix the parser.",
    )


def test_create_run_persists_state_and_task_artifact(tmp_path) -> None:
    store = RunStore(tmp_path / "artifacts")
    created = store.create_run(task())

    restored = RunStore(tmp_path / "artifacts").get_run(created.run_id)
    artifacts = store.list_artifacts(created.run_id)

    assert restored == created
    assert [path.name for path in artifacts] == ["task.json"]
    assert '"task_id": "issue-1"' in artifacts[0].read_text(encoding="utf-8")


def test_stage_progression_is_explicit(tmp_path) -> None:
    store = RunStore(tmp_path)
    state = store.create_run(task())

    started = store.start_stage(state.run_id, RunStage.INTAKE)
    completed = store.complete_stage(started.run_id, artifact_paths=[Path("task.json")])

    assert started.status is RunStatus.RUNNING
    assert completed.status is RunStatus.PENDING
    assert completed.current_stage is RunStage.INVESTIGATION
    assert completed.stages[0].status is RunStatus.COMPLETED
    assert completed.stages[0].artifact_paths == [Path("task.json")]


def test_invalid_stage_transition_is_rejected(tmp_path) -> None:
    store = RunStore(tmp_path)
    state = store.create_run(task())

    with pytest.raises(InvalidTransitionError, match="expected stage intake"):
        store.start_stage(state.run_id, RunStage.PLANNING)

    store.start_stage(state.run_id)
    with pytest.raises(InvalidTransitionError, match="while run is running"):
        store.start_stage(state.run_id)


def test_resume_preserves_interrupted_attempt_and_retries_current_stage(tmp_path) -> None:
    store = RunStore(tmp_path)
    state = store.create_run(task())
    store.start_stage(state.run_id)

    resumed = store.resume_run(state.run_id)
    restarted = store.start_stage(state.run_id)

    assert resumed.status is RunStatus.PENDING
    assert resumed.current_stage is RunStage.INTAKE
    assert resumed.stages[0].status is RunStatus.FAILED
    assert "Interrupted" in (resumed.stages[0].error or "")
    assert len(restarted.stages) == 2
    assert restarted.stages[-1].status is RunStatus.RUNNING


def test_failed_stage_can_be_resumed(tmp_path) -> None:
    store = RunStore(tmp_path)
    state = store.create_run(task())
    store.start_stage(state.run_id)
    failed = store.fail_stage(state.run_id, "provider unavailable")

    resumed = store.resume_run(state.run_id)

    assert failed.status is RunStatus.FAILED
    assert resumed.status is RunStatus.PENDING
    assert resumed.current_stage is RunStage.INTAKE


def test_artifact_paths_cannot_escape_run_directory(tmp_path) -> None:
    store = RunStore(tmp_path)
    state = store.create_run(task())

    with pytest.raises(UnsafeArtifactPathError):
        store.write_artifact(state.run_id, "../outside.json", {})


def test_completed_run_cannot_resume(tmp_path) -> None:
    store = RunStore(tmp_path)
    state = store.create_run(task())

    for stage in RunStage:
        store.start_stage(state.run_id, stage)
        state = store.complete_stage(state.run_id)

    assert state.status is RunStatus.COMPLETED
    with pytest.raises(InvalidTransitionError, match="cannot be resumed"):
        store.resume_run(state.run_id)
