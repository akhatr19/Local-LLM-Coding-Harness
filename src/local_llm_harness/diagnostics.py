"""Local dependency and service checks used by the doctor command."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from local_llm_harness.config import HarnessSettings
from local_llm_harness.indexing import SentenceTransformerEmbedder
from local_llm_harness.lockfile import verify_lockfile


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: str
    details: str


def collect_diagnostics(
    settings: HarnessSettings,
    *,
    project_root: Path,
    check_services: bool,
    check_embeddings: bool,
    strict: bool,
) -> list[DiagnosticCheck]:
    live_services = check_services or strict
    live_embeddings = check_embeddings or strict
    checks = [
        _python_check(),
        DiagnosticCheck("Configuration", "PASS", "valid"),
        _package_check("LiteLLM package", "litellm"),
        _package_check("ChromaDB package", "chromadb"),
        _package_check("Sentence Transformers package", "sentence_transformers"),
        _package_check("SWE-bench package", "swebench", optional=True),
        _command_check("Git", "git", strict=True),
        _command_check("ripgrep", "rg", strict=True),
        _storage_check(settings.artifacts.root),
        _lockfile_check(project_root),
        _image_reference_check(settings.docker.image),
    ]
    checks.extend(_docker_checks(settings, live=live_services, strict=strict))
    checks.append(_searxng_check(settings, live=live_services, strict=strict))
    checks.append(_embedding_check(settings, live=live_embeddings, strict=strict))
    return checks


def _python_check() -> DiagnosticCheck:
    supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    return DiagnosticCheck(
        "Python",
        "PASS" if supported else "FAIL",
        ".".join(map(str, sys.version_info[:3])),
    )


def _package_check(name: str, module: str, *, optional: bool = False) -> DiagnosticCheck:
    installed = importlib.util.find_spec(module) is not None
    status = "PASS" if installed else ("SKIP" if optional else "FAIL")
    return DiagnosticCheck(name, status, "installed" if installed else "missing")


def _command_check(name: str, command: str, *, strict: bool) -> DiagnosticCheck:
    path = shutil.which(command)
    status = "PASS" if path else ("FAIL" if strict else "WARN")
    return DiagnosticCheck(name, status, path or "missing")


def _storage_check(root: Path) -> DiagnosticCheck:
    parent = _nearest_existing_parent(root)
    try:
        with tempfile.NamedTemporaryFile(prefix="harness-doctor-", dir=parent):
            pass
    except OSError as exc:
        return DiagnosticCheck("Artifact storage", "FAIL", str(exc))
    return DiagnosticCheck("Artifact storage", "PASS", str(parent))


def _lockfile_check(project_root: Path) -> DiagnosticCheck:
    valid, details = verify_lockfile(project_root)
    return DiagnosticCheck("Dependency lock", "PASS" if valid else "FAIL", details)


def _image_reference_check(image: str) -> DiagnosticCheck:
    pinned = "@sha256:" in image and len(image.rsplit("@sha256:", 1)[1]) == 64
    return DiagnosticCheck(
        "Sandbox image reference",
        "PASS" if pinned else "FAIL",
        image if pinned else "image must include a sha256 digest",
    )


def _docker_checks(
    settings: HarnessSettings, *, live: bool, strict: bool
) -> list[DiagnosticCheck]:
    docker = shutil.which("docker")
    if docker is None:
        status = "FAIL" if strict else "WARN"
        return [
            DiagnosticCheck("Docker CLI", status, "missing"),
            DiagnosticCheck("Docker daemon", "SKIP", "Docker CLI is unavailable"),
            DiagnosticCheck("Sandbox image", "SKIP", "Docker CLI is unavailable"),
        ]
    checks = [DiagnosticCheck("Docker CLI", "PASS", docker)]
    if not live:
        checks.extend(
            [
                DiagnosticCheck("Docker daemon", "SKIP", "use --check-services"),
                DiagnosticCheck("Sandbox image", "SKIP", "use --check-services"),
            ]
        )
        return checks
    daemon = _run([docker, "info", "--format", "{{.ServerVersion}}"])
    checks.append(_command_result("Docker daemon", daemon, strict=strict))
    image = _run([docker, "image", "inspect", settings.docker.image])
    checks.append(_command_result("Sandbox image", image, strict=strict))
    return checks


def _searxng_check(
    settings: HarnessSettings, *, live: bool, strict: bool
) -> DiagnosticCheck:
    if not live:
        return DiagnosticCheck("SearXNG JSON search", "SKIP", "use --check-services")
    try:
        response = httpx.get(
            f"{settings.searxng.base_url.rstrip('/')}/search",
            params={"q": "python standard library", "format": "json"},
            timeout=settings.searxng.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("response does not contain a results list")
    except (httpx.HTTPError, ValueError) as exc:
        return DiagnosticCheck(
            "SearXNG JSON search", "FAIL" if strict else "WARN", str(exc)
        )
    return DiagnosticCheck("SearXNG JSON search", "PASS", settings.searxng.base_url)


def _embedding_check(
    settings: HarnessSettings, *, live: bool, strict: bool
) -> DiagnosticCheck:
    if not live:
        return DiagnosticCheck("Embedding model", "SKIP", "use --check-embeddings")
    try:
        vectors = SentenceTransformerEmbedder(settings.retrieval.embedding_model).embed(
            ["diagnostic probe"]
        )
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("embedding model returned no vector")
    except Exception as exc:
        return DiagnosticCheck("Embedding model", "FAIL" if strict else "WARN", str(exc))
    return DiagnosticCheck("Embedding model", "PASS", settings.retrieval.embedding_model)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 1, "", str(exc))


def _command_result(
    name: str, result: subprocess.CompletedProcess[str], *, strict: bool
) -> DiagnosticCheck:
    details = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    status = "PASS" if result.returncode == 0 else ("FAIL" if strict else "WARN")
    return DiagnosticCheck(name, status, details)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
