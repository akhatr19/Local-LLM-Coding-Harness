"""SQLite run metadata and filesystem artifact persistence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel

from local_llm_harness.contracts import (
    TERMINAL_FAILURE_STATUSES,
    RunStage,
    RunState,
    RunStatus,
    StageResult,
    TaskSpec,
    utc_now,
)

ArtifactModel = TypeVar("ArtifactModel", bound=BaseModel)

STAGE_ORDER = (
    RunStage.INTAKE,
    RunStage.INVESTIGATION,
    RunStage.PLANNING,
    RunStage.RESEARCH,
    RunStage.IMPLEMENTATION,
    RunStage.EVALUATION,
)


class RunStoreError(RuntimeError):
    """Base persistence error."""


class RunNotFoundError(RunStoreError):
    """Raised when a run ID does not exist."""


class InvalidTransitionError(RunStoreError):
    """Raised when a stage or status transition violates the workflow."""


class UnsafeArtifactPathError(RunStoreError):
    """Raised when an artifact path escapes its run directory."""


class RunStore:
    """Persist run state transactionally and stage artifacts as ordinary files."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.runs_root = self.root / "runs"
        self.database_path = self.root / "runs.sqlite3"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS runs_task_id_idx ON runs(task_id)")

    def create_run(self, task: TaskSpec) -> RunState:
        state = RunState(task_id=task.task_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, task_id, status, current_stage, state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(state.run_id),
                    state.task_id,
                    state.status.value,
                    state.current_stage.value,
                    state.model_dump_json(),
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                ),
            )
        self.write_artifact(state.run_id, "task.json", task)
        return state

    def get_run(self, run_id: UUID | str) -> RunState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (str(run_id),)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return RunState.model_validate_json(row["state_json"])

    def start_stage(self, run_id: UUID | str, stage: RunStage | None = None) -> RunState:
        state = self.get_run(run_id)
        requested_stage = stage or state.current_stage
        if state.status is not RunStatus.PENDING:
            raise InvalidTransitionError(
                f"cannot start {requested_stage.value} while run is {state.status.value}"
            )
        if requested_stage is not state.current_stage:
            raise InvalidTransitionError(
                f"expected stage {state.current_stage.value}, got {requested_stage.value}"
            )

        started_at = utc_now()
        state.status = RunStatus.RUNNING
        state.stages.append(
            StageResult(
                stage=requested_stage,
                status=RunStatus.RUNNING,
                started_at=started_at,
            )
        )
        state.updated_at = started_at
        self._save_state(state)
        return state

    def complete_stage(
        self,
        run_id: UUID | str,
        *,
        artifact_paths: list[Path] | None = None,
    ) -> RunState:
        state = self.get_run(run_id)
        active_index = self._active_stage_index(state)
        finished_at = utc_now()
        active = state.stages[active_index]
        state.stages[active_index] = active.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "finished_at": finished_at,
                "artifact_paths": artifact_paths or [],
            }
        )

        next_stage = self._next_stage(state.current_stage)
        if next_stage is None:
            state.status = RunStatus.COMPLETED
        else:
            state.current_stage = next_stage
            state.status = RunStatus.PENDING
        state.updated_at = finished_at
        self._save_state(state)
        return state

    def fail_stage(
        self,
        run_id: UUID | str,
        error: str,
        *,
        status: RunStatus = RunStatus.FAILED,
    ) -> RunState:
        if status not in TERMINAL_FAILURE_STATUSES:
            raise InvalidTransitionError(f"{status.value} is not a terminal failure status")
        state = self.get_run(run_id)
        active_index = self._active_stage_index(state)
        finished_at = utc_now()
        state.stages[active_index] = state.stages[active_index].model_copy(
            update={"status": status, "finished_at": finished_at, "error": error}
        )
        state.status = status
        state.updated_at = finished_at
        self._save_state(state)
        return state

    def resume_run(self, run_id: UUID | str) -> RunState:
        """Reset an interrupted or failed current stage without losing its attempt history."""

        state = self.get_run(run_id)
        if state.status is RunStatus.COMPLETED:
            raise InvalidTransitionError("a completed run cannot be resumed")
        now = utc_now()
        for index in range(len(state.stages) - 1, -1, -1):
            attempt = state.stages[index]
            if attempt.stage is state.current_stage and attempt.status is RunStatus.RUNNING:
                state.stages[index] = attempt.model_copy(
                    update={
                        "status": RunStatus.FAILED,
                        "finished_at": now,
                        "error": "Interrupted before stage completion",
                    }
                )
                break
        state.status = RunStatus.PENDING
        state.updated_at = now
        self._save_state(state)
        return state

    def write_artifact(
        self,
        run_id: UUID | str,
        relative_path: str,
        content: BaseModel | dict[str, Any] | list[Any] | str | bytes,
    ) -> Path:
        self.get_run(run_id)
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise UnsafeArtifactPathError("artifact path must remain inside the run directory")

        run_directory = self.runs_root / str(run_id)
        target = run_directory.joinpath(*pure_path.parts).resolve()
        if not target.is_relative_to(run_directory.resolve()):
            raise UnsafeArtifactPathError("artifact path must remain inside the run directory")
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            payload = content.model_dump(mode="json") if isinstance(content, BaseModel) else content
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return target

    def list_artifacts(self, run_id: UUID | str) -> list[Path]:
        self.get_run(run_id)
        run_directory = self.runs_root / str(run_id)
        if not run_directory.exists():
            return []
        return sorted(path for path in run_directory.rglob("*") if path.is_file())

    def read_artifact_model(
        self,
        run_id: UUID | str,
        relative_path: str,
        model: type[ArtifactModel],
    ) -> ArtifactModel:
        """Load and validate a JSON artifact stored inside a run directory."""

        self.get_run(run_id)
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise UnsafeArtifactPathError("artifact path must remain inside the run directory")
        run_directory = (self.runs_root / str(run_id)).resolve()
        target = run_directory.joinpath(*pure_path.parts).resolve()
        if not target.is_relative_to(run_directory):
            raise UnsafeArtifactPathError("artifact path must remain inside the run directory")
        try:
            return model.model_validate_json(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RunStoreError(f"artifact does not exist: {relative_path}") from exc

    def _save_state(self, state: RunState) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, current_stage = ?, state_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    state.status.value,
                    state.current_stage.value,
                    state.model_dump_json(),
                    state.updated_at.isoformat(),
                    str(state.run_id),
                ),
            )
        if cursor.rowcount != 1:
            raise RunNotFoundError(f"run not found: {state.run_id}")

    @staticmethod
    def _active_stage_index(state: RunState) -> int:
        if state.status is not RunStatus.RUNNING:
            raise InvalidTransitionError(
                f"run has no active stage while status is {state.status.value}"
            )
        for index in range(len(state.stages) - 1, -1, -1):
            attempt = state.stages[index]
            if attempt.stage is state.current_stage and attempt.status is RunStatus.RUNNING:
                return index
        raise InvalidTransitionError(f"no active attempt for {state.current_stage.value}")

    @staticmethod
    def _next_stage(stage: RunStage) -> RunStage | None:
        index = STAGE_ORDER.index(stage)
        if index == len(STAGE_ORDER) - 1:
            return None
        return STAGE_ORDER[index + 1]
