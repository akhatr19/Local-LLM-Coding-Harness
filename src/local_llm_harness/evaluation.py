"""Resumable baseline-versus-pipeline benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_llm_harness.config import DockerSettings, EvaluationSettings, ModelProfile
from local_llm_harness.contracts import EvaluationResult, utc_now


class EvaluationError(RuntimeError):
    """An evaluation could not be created, resumed, or completed."""


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    swebench_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    instances: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def instances_must_be_unique(self) -> BenchmarkManifest:
        if len(set(self.instances)) != len(self.instances):
            raise ValueError("benchmark instance IDs must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> BenchmarkManifest:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise EvaluationError(f"unable to load benchmark manifest: {exc}") from exc
        return cls.model_validate(payload)

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class EvaluationBudget(BaseModel):
    """Every compared mode receives one identical immutable resource envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    model_timeout_seconds: float
    max_tokens_per_call: int
    max_model_calls: int
    max_total_tokens: int
    run_timeout_seconds: int
    command_timeout_seconds: int
    max_commands: int
    max_agent_steps: int
    cpu_limit: float
    memory_mb: int
    pids_limit: int
    disk_mb: int
    tmpfs_mb: int
    network_disabled: bool

    @classmethod
    def from_settings(
        cls,
        profile: ModelProfile,
        docker: DockerSettings,
        evaluation: EvaluationSettings,
    ) -> EvaluationBudget:
        return cls(
            model=profile.model,
            model_timeout_seconds=profile.timeout_seconds,
            max_tokens_per_call=profile.max_tokens,
            max_model_calls=evaluation.max_model_calls,
            max_total_tokens=evaluation.max_total_tokens,
            run_timeout_seconds=docker.run_timeout_seconds,
            command_timeout_seconds=docker.command_timeout_seconds,
            max_commands=docker.max_commands,
            max_agent_steps=docker.max_agent_steps,
            cpu_limit=docker.cpu_limit,
            memory_mb=docker.memory_mb,
            pids_limit=docker.pids_limit,
            disk_mb=docker.disk_mb,
            tmpfs_mb=docker.tmpfs_mb,
            network_disabled=docker.network_disabled,
        )

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: EvaluationResult
    budget_hash: str = Field(pattern="^[0-9a-f]{64}$")
    finished_at: datetime = Field(default_factory=utc_now)

    @property
    def key(self) -> str:
        return f"{self.result.instance_id}:{self.result.mode}"


class EvaluationRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: UUID = Field(default_factory=uuid4)
    manifest: BenchmarkManifest
    manifest_hash: str = Field(pattern="^[0-9a-f]{64}$")
    model_profile: str
    modes: tuple[str, ...]
    budget: EvaluationBudget
    records: list[EvaluationRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_identity_and_records(self) -> EvaluationRun:
        if self.manifest_hash != self.manifest.digest:
            raise ValueError("manifest hash does not match manifest content")
        if not self.modes or any(mode not in {"baseline", "full"} for mode in self.modes):
            raise ValueError("evaluation modes must contain baseline and/or full")
        keys = [record.key for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("evaluation records must be unique per instance and mode")
        allowed = {
            f"{instance_id}:{mode}"
            for instance_id in self.manifest.instances
            for mode in self.modes
        }
        if not set(keys).issubset(allowed):
            raise ValueError("evaluation record does not belong to this run")
        if any(record.budget_hash != self.budget.digest for record in self.records):
            raise ValueError("compared modes did not receive the same resource budget")
        return self

    @property
    def expected_record_count(self) -> int:
        return len(self.manifest.instances) * len(self.modes)

    @property
    def complete(self) -> bool:
        return len(self.records) == self.expected_record_count


class ModeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str
    attempts: int
    resolved: int
    resolution_rate: float
    prompt_tokens: int
    completion_tokens: int
    model_calls: int
    duration_seconds: float
    research_requests: int
    failures: int


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: UUID
    complete: bool
    metrics: tuple[ModeMetrics, ...]
    records: tuple[EvaluationRecord, ...]

    @classmethod
    def from_run(cls, run: EvaluationRun) -> EvaluationReport:
        metrics = []
        for mode in run.modes:
            results = [record.result for record in run.records if record.result.mode == mode]
            resolved = sum(result.resolved for result in results)
            attempts = len(results)
            metrics.append(
                ModeMetrics(
                    mode=mode,
                    attempts=attempts,
                    resolved=resolved,
                    resolution_rate=resolved / attempts if attempts else 0,
                    prompt_tokens=sum(result.prompt_tokens for result in results),
                    completion_tokens=sum(result.completion_tokens for result in results),
                    model_calls=sum(result.model_calls for result in results),
                    duration_seconds=sum(result.duration_seconds for result in results),
                    research_requests=sum(result.research_requests for result in results),
                    failures=sum(result.failure_reason is not None for result in results),
                )
            )
        return cls(
            evaluation_id=run.evaluation_id,
            complete=run.complete,
            metrics=tuple(metrics),
            records=tuple(run.records),
        )


class EvaluationRunner(Protocol):
    async def run_instance(
        self,
        instance_id: str,
        mode: str,
        model_profile: str,
        budget: EvaluationBudget,
        evaluation_dir: Path,
    ) -> EvaluationResult: ...


class EvaluationStore:
    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.expanduser().resolve() / "evaluations"

    def create(
        self,
        manifest: BenchmarkManifest,
        model_profile: str,
        modes: Sequence[str],
        budget: EvaluationBudget,
    ) -> EvaluationRun:
        run = EvaluationRun(
            manifest=manifest,
            manifest_hash=manifest.digest,
            model_profile=model_profile,
            modes=tuple(modes),
            budget=budget,
        )
        self._write(run)
        return run

    def load(self, evaluation_id: UUID | str) -> EvaluationRun:
        path = self.directory(evaluation_id) / "state.json"
        try:
            return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EvaluationError(f"evaluation does not exist: {evaluation_id}") from exc

    def save(self, run: EvaluationRun) -> None:
        run.updated_at = utc_now()
        self._write(run)

    def directory(self, evaluation_id: UUID | str) -> Path:
        return self.root / str(evaluation_id)

    def _write(self, run: EvaluationRun) -> None:
        directory = self.directory(run.evaluation_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "state.json"
        temporary = directory / ".state.json.tmp"
        temporary.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)


class EvaluationHarness:
    def __init__(self, store: EvaluationStore, runner: EvaluationRunner) -> None:
        self.store = store
        self.runner = runner

    async def run(
        self,
        manifest: BenchmarkManifest,
        model_profile: str,
        budget: EvaluationBudget,
        *,
        modes: Sequence[str] = ("baseline", "full"),
        evaluation_id: UUID | str | None = None,
    ) -> EvaluationRun:
        selected_modes = tuple(modes)
        if evaluation_id is None:
            run = self.store.create(manifest, model_profile, selected_modes, budget)
        else:
            run = self.store.load(evaluation_id)
            self._validate_resume(run, manifest, model_profile, selected_modes, budget)

        completed = {record.key for record in run.records}
        directory = self.store.directory(run.evaluation_id)
        for instance_id in manifest.instances:
            for mode in selected_modes:
                key = f"{instance_id}:{mode}"
                if key in completed:
                    continue
                started = utc_now()
                try:
                    result = await self.runner.run_instance(
                        instance_id,
                        mode,
                        model_profile,
                        budget,
                        directory,
                    )
                    self._validate_result(result, instance_id, mode, model_profile)
                except Exception as exc:
                    result = EvaluationResult(
                        instance_id=instance_id,
                        model_profile=model_profile,
                        mode=mode,
                        resolved=False,
                        duration_seconds=(utc_now() - started).total_seconds(),
                        failure_reason=str(exc),
                    )
                run.records.append(EvaluationRecord(result=result, budget_hash=budget.digest))
                self.store.save(run)
                self.write_reports(run)
        return run

    def write_reports(self, run: EvaluationRun) -> None:
        report = EvaluationReport.from_run(run)
        directory = self.store.directory(run.evaluation_id)
        (directory / "results.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (directory / "comparison.md").write_text(_markdown_report(run, report), encoding="utf-8")

    @staticmethod
    def _validate_resume(
        run: EvaluationRun,
        manifest: BenchmarkManifest,
        profile: str,
        modes: tuple[str, ...],
        budget: EvaluationBudget,
    ) -> None:
        if run.manifest_hash != manifest.digest:
            raise EvaluationError("resume manifest differs from the persisted evaluation")
        if run.model_profile != profile or run.modes != modes:
            raise EvaluationError("resume profile or modes differ from the persisted evaluation")
        if run.budget.digest != budget.digest:
            raise EvaluationError("resume resource budget differs from the persisted evaluation")

    @staticmethod
    def _validate_result(
        result: EvaluationResult,
        instance_id: str,
        mode: str,
        profile: str,
    ) -> None:
        if (result.instance_id, result.mode, result.model_profile) != (
            instance_id,
            mode,
            profile,
        ):
            raise EvaluationError("evaluation runner returned mismatched result identity")


def _markdown_report(run: EvaluationRun, report: EvaluationReport) -> str:
    rows = [
        "# SWE-bench comparison",
        "",
        f"Evaluation: `{run.evaluation_id}`",
        f"Manifest: `{run.manifest.name}` ({len(run.manifest.instances)} instances)",
        f"Model profile: `{run.model_profile}`",
        f"Status: `{'complete' if report.complete else 'in progress'}`",
        "",
        "| Mode | Resolved | Rate | Tokens | Calls | Latency (s) | Research | Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in report.metrics:
        rows.append(
            f"| {metric.mode} | {metric.resolved}/{metric.attempts} | "
            f"{metric.resolution_rate:.1%} | "
            f"{metric.prompt_tokens + metric.completion_tokens} | {metric.model_calls} | "
            f"{metric.duration_seconds:.2f} | {metric.research_requests} | {metric.failures} |"
        )
    rows.extend(
        [
            "",
            f"All modes used the same immutable resource-budget hash: `{run.budget.digest}`.",
            "",
            "## Instance results",
            "",
            "| Instance | Mode | Resolved | Failure |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in report.records:
        result = record.result
        failure = (result.failure_reason or "").replace("|", "\\|").replace("\n", " ")
        rows.append(
            f"| {result.instance_id} | {result.mode} | "
            f"{'yes' if result.resolved else 'no'} | {failure} |"
        )
    return "\n".join(rows) + "\n"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
