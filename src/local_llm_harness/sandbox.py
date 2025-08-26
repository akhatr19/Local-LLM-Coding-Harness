"""Disposable repository copies and resource-bounded Docker command execution."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from local_llm_harness.config import DockerSettings


class SandboxError(RuntimeError):
    """A disposable workspace or Docker boundary failed."""


class SandboxCommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    disk_limit_exceeded: bool = False


@dataclass(frozen=True)
class DisposableWorkspace:
    root: Path
    source_repository: Path
    source_commit: str

    def changed_files(self) -> list[str]:
        tracked = _run_git(
            self.root,
            ["diff", "--name-only", "--relative", "HEAD", "--"],
        ).stdout.splitlines()
        untracked = _run_git(
            self.root,
            ["ls-files", "--others", "--exclude-standard"],
        ).stdout.splitlines()
        return sorted(set(tracked + untracked))

    def patch(self) -> str:
        return _run_git(
            self.root,
            ["diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        ).stdout

    def verify_patch_against_source(self, patch: str) -> None:
        current_commit = _run_git(self.source_repository, ["rev-parse", "HEAD"]).stdout.strip()
        if current_commit != self.source_commit:
            raise SandboxError("source repository revision changed during implementation")
        if _run_git(self.source_repository, ["status", "--porcelain"]).stdout:
            raise SandboxError("source repository changed during implementation")
        result = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=self.source_repository,
            input=patch,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise SandboxError(f"final patch does not apply cleanly: {result.stderr.strip()}")


class WorkspaceBuilder:
    """Export a clean Git revision without sharing the host repository metadata."""

    def create(
        self,
        source_repository: Path,
        destination: Path,
        *,
        expected_commit: str | None = None,
    ) -> DisposableWorkspace:
        source = source_repository.expanduser().resolve()
        destination = destination.expanduser().resolve()
        if not source.is_dir():
            raise SandboxError(f"source repository does not exist: {source}")
        if destination.exists():
            raise SandboxError(f"sandbox workspace already exists: {destination}")
        if _run_git(source, ["status", "--porcelain"]).stdout:
            raise SandboxError("source repository must be clean before implementation")
        source_commit = _run_git(source, ["rev-parse", "HEAD"]).stdout.strip()
        if expected_commit is not None and source_commit != expected_commit:
            raise SandboxError("source repository no longer matches the task base revision")
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive_path = destination.parent / f"{destination.name}-{uuid4().hex}.tar"
        try:
            _run_git(
                source,
                ["archive", "--format=tar", f"--output={archive_path}", source_commit],
            )
            destination.mkdir()
            with tarfile.open(archive_path, mode="r") as archive:
                for member in archive.getmembers():
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(destination):
                        raise SandboxError("Git archive contains an unsafe path")
                archive.extractall(destination)
        finally:
            archive_path.unlink(missing_ok=True)

        _run_git(destination, ["init", "-b", "sandbox"])
        _run_git(destination, ["add", "--all"])
        _run_git(
            destination,
            [
                "-c",
                "user.name=Harness Sandbox",
                "-c",
                "user.email=sandbox@localhost",
                "commit",
                "--quiet",
                "-m",
                "sandbox baseline",
            ],
        )
        return DisposableWorkspace(
            root=destination,
            source_repository=source,
            source_commit=source_commit,
        )


class DockerSandbox:
    """Run commands in isolated, non-root, networkless disposable containers."""

    def __init__(self, settings: DockerSettings, workspace: Path) -> None:
        self.settings = settings
        self.workspace = workspace.expanduser().resolve()
        if not settings.network_disabled:
            raise SandboxError("implementation container networking must remain disabled")
        host_uid = os.getuid() if hasattr(os, "getuid") else 65532
        host_gid = os.getgid() if hasattr(os, "getgid") else 65532
        self.user_id = host_uid if host_uid > 0 else 65532
        self.group_id = host_gid if host_gid > 0 else 65532

    def check_ready(self) -> None:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.settings.image],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"Docker is unavailable: {exc}") from exc
        if result.returncode != 0:
            raise SandboxError(
                f"sandbox image is unavailable: {self.settings.image}: {result.stderr.strip()}"
            )

    async def run(self, command: tuple[str, ...]) -> SandboxCommandResult:
        if not command or any(not argument or "\0" in argument for argument in command):
            raise SandboxError("sandbox command arguments must be non-empty")
        if _directory_size(self.workspace) > self.settings.disk_mb * 1024 * 1024:
            raise SandboxError("sandbox workspace already exceeds its disk limit")

        container_name = f"harness-{uuid4().hex[:20]}"
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--user",
            f"{self.user_id}:{self.group_id}",
            "--cpus",
            str(self.settings.cpu_limit),
            "--memory",
            f"{self.settings.memory_mb}m",
            "--memory-swap",
            f"{self.settings.memory_mb}m",
            "--pids-limit",
            str(self.settings.pids_limit),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--ulimit",
            "nofile=1024:1024",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={self.settings.tmpfs_mb}m",
            "--mount",
            f"type=bind,source={self.workspace},target=/workspace",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            self.settings.image,
            *command,
        ]
        started = monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise SandboxError(f"failed to start Docker: {exc}") from exc

        communicate = asyncio.create_task(process.communicate())
        timed_out = False
        disk_exceeded = False
        deadline = asyncio.get_running_loop().time() + self.settings.command_timeout_seconds
        try:
            while not communicate.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    timed_out = True
                    await self._kill(container_name)
                    break
                await asyncio.wait({communicate}, timeout=min(0.25, remaining))
                if _directory_size(self.workspace) > self.settings.disk_mb * 1024 * 1024:
                    disk_exceeded = True
                    await self._kill(container_name)
                    break
            stdout_bytes, stderr_bytes = await communicate
        except BaseException:
            await self._kill(container_name)
            if not communicate.done():
                communicate.cancel()
            raise

        if _directory_size(self.workspace) > self.settings.disk_mb * 1024 * 1024:
            disk_exceeded = True

        return SandboxCommandResult(
            command=command,
            exit_code=process.returncode if process.returncode is not None else 125,
            stdout=_bounded_output(stdout_bytes),
            stderr=_bounded_output(stderr_bytes),
            duration_seconds=monotonic() - started,
            timed_out=timed_out,
            disk_limit_exceeded=disk_exceeded,
        )

    @staticmethod
    async def _kill(container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
        except OSError:
            return


def _run_git(repository: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxError(f"Git operation failed: {exc}") from exc
    if result.returncode != 0:
        raise SandboxError(f"Git operation failed: {result.stderr.strip()}")
    return result


def _directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _bounded_output(value: bytes, limit: int = 1_000_000) -> str:
    if len(value) <= limit:
        return value.decode("utf-8", errors="replace")
    marker = b"\n... output truncated by harness ...\n"
    retained = value[: max(0, limit - len(marker))] + marker
    return retained.decode("utf-8", errors="replace")
