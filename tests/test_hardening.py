from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_llm_harness.config import LoggingSettings
from local_llm_harness.contracts import RunStage, TaskSpec
from local_llm_harness.lockfile import verify_lockfile
from local_llm_harness.observability import configure_logging
from local_llm_harness.storage import RunNotFoundError, RunStore


def test_repository_lockfile_obeys_cutoff() -> None:
    valid, details = verify_lockfile(Path.cwd())

    assert valid, details


def test_log_path_cannot_escape_artifact_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="inside the artifact"):
        configure_logging(LoggingSettings(json_file="../outside.jsonl"), tmp_path)


def test_retention_preview_and_apply(tmp_path) -> None:
    store = RunStore(tmp_path / "artifacts")
    state = store.create_run(
        TaskSpec(task_id="old", repository=tmp_path, problem_statement="fixture")
    )
    for stage in RunStage:
        store.start_stage(state.run_id, stage)
        state = store.complete_stage(state.run_id)
    cutoff = datetime.now(UTC) + timedelta(seconds=1)

    preview = store.prune_runs(
        cutoff=cutoff,
        max_completed_runs=100,
        retain_failed_runs=True,
    )
    applied = store.prune_runs(
        cutoff=cutoff,
        max_completed_runs=100,
        retain_failed_runs=True,
        dry_run=False,
    )

    assert preview.selected == (state.run_id,)
    assert preview.removed == ()
    assert applied.removed == (state.run_id,)
    with pytest.raises(RunNotFoundError):
        store.get_run(state.run_id)
