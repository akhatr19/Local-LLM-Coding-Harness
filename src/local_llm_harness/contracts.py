"""Pydantic contracts exchanged by workflow stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSpec(ContractModel):
    task_id: str = Field(min_length=1)
    repository: Path
    problem_statement: str = Field(min_length=1)
    base_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRef(ContractModel):
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt: str = ""
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_location(self) -> EvidenceRef:
        path = PurePosixPath(self.file_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence file_path must be repository-relative")
        if self.end_line < self.start_line:
            raise ValueError("evidence end_line must not precede start_line")
        return self


class InvestigationTopic(ContractModel):
    topic_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    search_terms: list[str] = Field(min_length=1)


class InvestigationTopicSet(ContractModel):
    topics: list[InvestigationTopic] = Field(min_length=1, max_length=4)


class InvestigatorReport(ContractModel):
    topic_id: str
    summary: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class InvestigationReport(ContractModel):
    task_id: str
    summary: str = Field(min_length=1)
    components: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    is_clear: bool = False


class PlanStep(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    order: int = Field(ge=1)
    description: str = Field(min_length=1)
    target_files: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()


class PlanCandidate(ContractModel):
    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def steps_must_be_sequential(self) -> PlanCandidate:
        orders = [step.order for step in self.steps]
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError("plan step order must be sequential starting at 1")
        return self


class PlanScore(ContractModel):
    candidate_id: str
    judge_id: str
    correctness: int = Field(ge=1, le=5)
    repository_fit: int = Field(ge=1, le=5)
    testability: int = Field(ge=1, le=5)
    risk: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    explanation: str = Field(min_length=1)

    @property
    def total(self) -> int:
        return (
            self.correctness
            + self.repository_fit
            + self.testability
            + self.risk
            + self.completeness
        )


class ResearchFinding(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_url: HttpUrl
    source_title: str = Field(min_length=1)
    relevance: str = Field(min_length=1)


class ResearchQuerySet(ContractModel):
    queries: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def queries_must_be_unique(self) -> ResearchQuerySet:
        if any(not query.strip() for query in self.queries):
            raise ValueError("research queries cannot be blank")
        if len(set(self.queries)) != len(self.queries):
            raise ValueError("research queries must be unique")
        return self


class ResearchReport(ContractModel):
    findings: list[ResearchFinding] = Field(min_length=1)
    conflicts: list[str] = Field(default_factory=list)


class FinalPlan(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    selected_candidate_id: str
    title: str = Field(min_length=1)
    steps: tuple[PlanStep, ...] = Field(min_length=1)
    research: tuple[ResearchFinding, ...] = ()
    plan_hash: str = Field(pattern="^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        selected_candidate_id: str,
        title: str,
        steps: Sequence[PlanStep],
        research: Sequence[ResearchFinding],
    ) -> Self:
        payload = cls._hash_payload(
            task_id=task_id,
            selected_candidate_id=selected_candidate_id,
            title=title,
            steps=steps,
            research=research,
        )
        return cls(
            task_id=task_id,
            selected_candidate_id=selected_candidate_id,
            title=title,
            steps=tuple(steps),
            research=tuple(research),
            plan_hash=cls._digest(payload),
        )

    @model_validator(mode="after")
    def hash_must_match_content(self) -> FinalPlan:
        payload = self._hash_payload(
            task_id=self.task_id,
            selected_candidate_id=self.selected_candidate_id,
            title=self.title,
            steps=self.steps,
            research=self.research,
        )
        if self.plan_hash != self._digest(payload):
            raise ValueError("plan_hash does not match final plan content")
        return self

    @staticmethod
    def _hash_payload(
        *,
        task_id: str,
        selected_candidate_id: str,
        title: str,
        steps: Sequence[PlanStep],
        research: Sequence[ResearchFinding],
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "selected_candidate_id": selected_candidate_id,
            "title": title,
            "steps": [step.model_dump(mode="json") for step in steps],
            "research": [finding.model_dump(mode="json") for finding in research],
        }

    @staticmethod
    def _digest(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class RunStage(StrEnum):
    INTAKE = "intake"
    INVESTIGATION = "investigation"
    PLANNING = "planning"
    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PLAN_BLOCKED = "plan_blocked"
    RESEARCH_FAILED = "research_failed"
    SANDBOX_FAILED = "sandbox_failed"


TERMINAL_FAILURE_STATUSES = frozenset(
    {
        RunStatus.FAILED,
        RunStatus.PLAN_BLOCKED,
        RunStatus.RESEARCH_FAILED,
        RunStatus.SANDBOX_FAILED,
    }
)


class StageResult(ContractModel):
    stage: RunStage
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    artifact_paths: list[Path] = Field(default_factory=list)
    error: str | None = None


class RunState(ContractModel):
    run_id: UUID = Field(default_factory=uuid4)
    task_id: str
    status: RunStatus = RunStatus.PENDING
    current_stage: RunStage = RunStage.INTAKE
    stages: list[StageResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EvaluationResult(ContractModel):
    instance_id: str
    model_profile: str
    mode: str = Field(pattern="^(baseline|full)$")
    resolved: bool
    duration_seconds: float = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    research_requests: int = Field(default=0, ge=0)
    failure_reason: str | None = None
