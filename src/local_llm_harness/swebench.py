"""SWE-bench 4.0.3 dataset, solver, and official grading adapters."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Any

from local_llm_harness.baseline import DirectCodingAgent
from local_llm_harness.config import HarnessSettings
from local_llm_harness.contracts import EvaluationResult, TaskSpec
from local_llm_harness.evaluation import EvaluationBudget, EvaluationError
from local_llm_harness.model_gateway import BudgetedModelGateway, LiteLLMGateway
from local_llm_harness.pipeline import run_full_pipeline
from local_llm_harness.research import SearxNGClient


class CountingSearxNGClient(SearxNGClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests = 0

    async def search_and_fetch(self, queries):
        self.requests += len(queries)
        return await super().search_and_fetch(queries)


class SWEbenchDataset:
    def __init__(self, dataset_name: str, split: str, expected_version: str) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.expected_version = expected_version
        self._records: dict[str, dict[str, Any]] | None = None

    def record(self, instance_id: str) -> dict[str, Any]:
        if self._records is None:
            self._records = self._load()
        try:
            return self._records[instance_id]
        except KeyError as exc:
            raise EvaluationError(
                f"instance is not in the configured dataset: {instance_id}"
            ) from exc

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            installed = importlib.metadata.version("swebench")
        except importlib.metadata.PackageNotFoundError as exc:
            raise EvaluationError(
                "SWE-bench support is not installed; install the 'swebench' extra"
            ) from exc
        if installed != self.expected_version:
            raise EvaluationError(
                f"SWE-bench {self.expected_version} is required, found {installed}"
            )
        try:
            from swebench.harness.utils import load_swebench_dataset

            records = load_swebench_dataset(self.dataset_name, self.split)
        except Exception as exc:
            raise EvaluationError(f"unable to load SWE-bench dataset: {exc}") from exc
        return {str(record["instance_id"]): dict(record) for record in records}


class SWEbenchRepositoryProvider:
    def prepare(self, record: dict[str, Any], root: Path) -> Path:
        instance_id = str(record["instance_id"])
        repository_name = str(record["repo"])
        base_commit = str(record["base_commit"])
        destination = root / instance_id
        if destination.exists():
            commit = self._git(destination, "rev-parse", "HEAD").stdout.strip()
            status = self._git(destination, "status", "--porcelain").stdout
            if commit != base_commit or status:
                raise EvaluationError(f"cached repository is invalid for {instance_id}")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                f"https://github.com/{repository_name}.git",
                str(destination),
            ],
            cwd=root,
            timeout=1800,
        )
        self._git(destination, "checkout", "--quiet", base_commit)
        return destination

    @classmethod
    def _git(cls, repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return cls._run(["git", *arguments], cwd=repository, timeout=300)

    @staticmethod
    def _run(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvaluationError(f"command failed to start: {exc}") from exc
        if result.returncode != 0:
            raise EvaluationError(
                f"command failed ({result.returncode}): {' '.join(command)}\n"
                f"{result.stderr.strip()}"
            )
        return result


class OfficialSWEbenchGrader:
    """Grade one prediction with the pinned upstream evaluation harness."""

    def __init__(
        self,
        *,
        dataset_name: str,
        split: str,
        timeout_seconds: int,
        cache_level: str,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.timeout_seconds = timeout_seconds
        self.cache_level = cache_level

    def grade(
        self,
        *,
        instance_id: str,
        model_name: str,
        patch: str,
        output_dir: Path,
    ) -> bool:
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / "predictions.jsonl"
        predictions.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": model_name,
                    "model_patch": patch,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run_id = f"{output_dir.parent.name}-{output_dir.name}"
        SWEbenchRepositoryProvider._run(
            [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                self.dataset_name,
                "--split",
                self.split,
                "--predictions_path",
                str(predictions),
                "--max_workers",
                "1",
                "--run_id",
                run_id,
                "--cache_level",
                self.cache_level,
                "--instance_ids",
                instance_id,
            ],
            cwd=output_dir,
            timeout=self.timeout_seconds,
        )
        resolved = _find_resolution(output_dir, instance_id)
        if resolved is None:
            raise EvaluationError("official SWE-bench report did not contain an instance result")
        return resolved


class SWEbenchComparisonRunner:
    """Generate and officially grade baseline/full patches under equal budgets."""

    def __init__(self, settings: HarnessSettings) -> None:
        self.settings = settings
        evaluation = settings.evaluation
        self.dataset = SWEbenchDataset(
            evaluation.dataset_name,
            evaluation.split,
            evaluation.swebench_version,
        )
        self.repositories = SWEbenchRepositoryProvider()
        self.grader = OfficialSWEbenchGrader(
            dataset_name=evaluation.dataset_name,
            split=evaluation.split,
            timeout_seconds=evaluation.grading_timeout_seconds,
            cache_level=evaluation.cache_level,
        )

    async def run_instance(
        self,
        instance_id: str,
        mode: str,
        model_profile: str,
        budget: EvaluationBudget,
        evaluation_dir: Path,
    ) -> EvaluationResult:
        started = monotonic()
        gateway = BudgetedModelGateway(
            LiteLLMGateway(self.settings.litellm),
            max_calls=budget.max_model_calls,
            max_total_tokens=budget.max_total_tokens,
        )
        research_counter = [0]
        try:
            return await self._run_instance(
                instance_id=instance_id,
                mode=mode,
                model_profile=model_profile,
                budget=budget,
                evaluation_dir=evaluation_dir,
                gateway=gateway,
                research_counter=research_counter,
                started=started,
            )
        except Exception as exc:
            return EvaluationResult(
                instance_id=instance_id,
                model_profile=model_profile,
                mode=mode,
                resolved=False,
                duration_seconds=monotonic() - started,
                prompt_tokens=gateway.usage.prompt_tokens,
                completion_tokens=gateway.usage.completion_tokens,
                model_calls=gateway.calls,
                research_requests=research_counter[0],
                failure_reason=str(exc),
            )

    async def _run_instance(
        self,
        *,
        instance_id: str,
        mode: str,
        model_profile: str,
        budget: EvaluationBudget,
        evaluation_dir: Path,
        gateway: BudgetedModelGateway,
        research_counter: list[int],
        started: float,
    ) -> EvaluationResult:
        expected = EvaluationBudget.from_settings(
            self.settings.litellm.profiles[model_profile],
            self.settings.docker,
            self.settings.evaluation,
        )
        if budget.digest != expected.digest:
            raise EvaluationError("runner resource budget does not match configured limits")
        record = await asyncio.to_thread(self.dataset.record, instance_id)
        repository = await asyncio.to_thread(
            self.repositories.prepare,
            record,
            evaluation_dir / "repositories",
        )
        async with asyncio.timeout(budget.run_timeout_seconds):
            coding, research_requests = await self._generate_patch(
                record=record,
                repository=repository,
                instance_id=instance_id,
                mode=mode,
                model_profile=model_profile,
                gateway=gateway,
                evaluation_dir=evaluation_dir,
                research_counter=research_counter,
            )

        grading_dir = evaluation_dir / "grading" / instance_id / mode
        resolved = await asyncio.to_thread(
            self.grader.grade,
            instance_id=instance_id,
            model_name=f"{model_profile}-{mode}",
            patch=coding.patch,
            output_dir=grading_dir,
        )
        return EvaluationResult(
            instance_id=instance_id,
            model_profile=model_profile,
            mode=mode,
            resolved=resolved,
            duration_seconds=monotonic() - started,
            prompt_tokens=gateway.usage.prompt_tokens,
            completion_tokens=gateway.usage.completion_tokens,
            model_calls=gateway.calls,
            research_requests=research_requests,
            failure_reason=None,
        )

    async def _generate_patch(
        self,
        *,
        record: dict[str, Any],
        repository: Path,
        instance_id: str,
        mode: str,
        model_profile: str,
        gateway: BudgetedModelGateway,
        evaluation_dir: Path,
        research_counter: list[int],
    ):
        if mode == "baseline":
            task = TaskSpec(
                task_id=instance_id,
                repository=repository,
                problem_statement=str(record["problem_statement"]),
                base_commit=str(record["base_commit"]),
                metadata={"dataset": self.settings.evaluation.dataset_name},
            )
            coding = await DirectCodingAgent(
                gateway=gateway,
                profile_name=model_profile,
                settings=self.settings.docker,
            ).run(task, evaluation_dir / "workspaces" / instance_id / mode)
            return coding, 0
        elif mode == "full":
            attempt_settings = self.settings.model_copy(
                update={
                    "artifacts": self.settings.artifacts.model_copy(
                        update={"root": evaluation_dir / "full-runs" / instance_id}
                    )
                }
            )
            research = CountingSearxNGClient(self.settings.searxng)
            try:
                outcome = await run_full_pipeline(
                    repository=repository,
                    issue=str(record["problem_statement"]),
                    profile=model_profile,
                    settings=attempt_settings,
                    gateway=gateway,
                    research_client=research,
                )
                coding = outcome.result
                research_requests = research.requests
            finally:
                research_counter[0] = research.requests
                await research.close()
            return coding, research_requests
        else:
            raise EvaluationError(f"unsupported evaluation mode: {mode}")


def _find_resolution(root: Path, instance_id: str) -> bool | None:
    for path in sorted(root.rglob("*.json")):
        if path.name == "predictions.jsonl":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            detail = payload.get(instance_id)
            if isinstance(detail, dict) and isinstance(detail.get("resolved"), bool):
                return detail["resolved"]
            for key in ("resolved_ids", "resolved_instances"):
                values = payload.get(key)
                if isinstance(values, list) and instance_id in values:
                    return True
            for key in ("unresolved_ids", "error_ids"):
                values = payload.get(key)
                if isinstance(values, list) and instance_id in values:
                    return False
    return None
