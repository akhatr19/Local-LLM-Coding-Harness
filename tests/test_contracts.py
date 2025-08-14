from pathlib import Path

import pytest
from pydantic import ValidationError

from local_llm_harness.contracts import (
    EvaluationResult,
    EvidenceRef,
    FinalPlan,
    PlanCandidate,
    PlanStep,
    RunState,
    TaskSpec,
)


def test_task_and_run_state_round_trip() -> None:
    task = TaskSpec(
        task_id="issue-1",
        repository=Path("/tmp/example"),
        problem_statement="Fix the parser.",
    )
    state = RunState(task_id=task.task_id)

    restored = RunState.model_validate_json(state.model_dump_json())

    assert restored == state


def test_evidence_must_be_relative_and_ordered() -> None:
    with pytest.raises(ValidationError, match="repository-relative"):
        EvidenceRef(
            file_path="../secret.txt",
            start_line=1,
            end_line=2,
            rationale="unsafe",
        )

    with pytest.raises(ValidationError, match="must not precede"):
        EvidenceRef(
            file_path="src/parser.py",
            start_line=10,
            end_line=2,
            rationale="reversed",
        )


def test_plan_steps_must_be_sequential() -> None:
    with pytest.raises(ValidationError, match="sequential"):
        PlanCandidate(
            candidate_id="plan-a",
            title="Parser fix",
            rationale="The failing branch is isolated.",
            steps=[PlanStep(order=2, description="Patch it")],
        )


def test_final_plan_is_immutable() -> None:
    plan = FinalPlan.create(
        task_id="issue-1",
        selected_candidate_id="plan-a",
        title="Parser fix",
        steps=(PlanStep(order=1, description="Patch it"),),
        research=(),
    )

    with pytest.raises(ValidationError, match="frozen"):
        plan.title = "Changed"  # type: ignore[misc]

    with pytest.raises(ValidationError, match="frozen"):
        plan.steps[0].description = "Changed"  # type: ignore[misc]

    assert isinstance(plan.steps[0].target_files, tuple)


def test_final_plan_rejects_a_mismatched_hash() -> None:
    with pytest.raises(ValidationError, match="plan_hash does not match"):
        FinalPlan(
            task_id="issue-1",
            selected_candidate_id="plan-a",
            title="Parser fix",
            steps=(PlanStep(order=1, description="Patch it"),),
            plan_hash="0" * 64,
        )


def test_evaluation_mode_is_constrained() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        EvaluationResult(
            instance_id="sample",
            model_profile="local",
            mode="experimental",
            resolved=False,
            duration_seconds=1,
        )
