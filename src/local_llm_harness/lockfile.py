"""Validate direct pins and the package upload-date cutoff."""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path

CUTOFF = datetime(2025, 7, 1, tzinfo=UTC)
EXPECTED_EXCLUDE_NEWER = "2025-07-01T00:00:00Z"


def verify_lockfile(project_root: Path) -> tuple[bool, str]:
    pyproject_path = project_root / "pyproject.toml"
    lock_path = project_root / "uv.lock"
    if not pyproject_path.is_file() or not lock_path.is_file():
        return False, "pyproject.toml or uv.lock is missing"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return False, f"unable to parse dependency metadata: {exc}"

    if lock.get("options", {}).get("exclude-newer") != EXPECTED_EXCLUDE_NEWER:
        return False, "uv.lock does not enforce the June 30, 2025 cutoff"
    unpinned = [
        dependency
        for dependency in _direct_dependencies(pyproject)
        if "==" not in dependency
    ]
    if unpinned:
        return False, "direct dependencies are not exact pins: " + ", ".join(unpinned)

    newest = None
    for package in lock.get("package", []):
        source = package.get("source", {})
        if "registry" not in source:
            continue
        files = [package.get("sdist"), *package.get("wheels", [])]
        for file in files:
            if not isinstance(file, dict) or "upload-time" not in file:
                continue
            uploaded = datetime.fromisoformat(file["upload-time"].replace("Z", "+00:00"))
            newest = uploaded if newest is None or uploaded > newest else newest
            if uploaded >= CUTOFF:
                name = package.get("name", "unknown")
                return False, f"{name} contains a file uploaded after the cutoff"
    if newest is None:
        return False, "uv.lock contains no registry upload timestamps"
    return True, f"exact pins; newest locked file: {newest.date().isoformat()}"


def _direct_dependencies(pyproject: dict) -> list[str]:
    dependencies = list(pyproject.get("build-system", {}).get("requires", []))
    project = pyproject.get("project", {})
    dependencies.extend(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        dependencies.extend(group)
    for group in pyproject.get("dependency-groups", {}).values():
        dependencies.extend(group)
    return dependencies
