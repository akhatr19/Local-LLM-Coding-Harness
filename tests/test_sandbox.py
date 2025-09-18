import os

import pytest

from local_llm_harness.config import DockerSettings
from local_llm_harness.sandbox import DockerSandbox, WorkspaceBuilder


def test_disposable_workspace_does_not_modify_source(sample_repository, tmp_path) -> None:
    original = (sample_repository / "src" / "parser.py").read_text(encoding="utf-8")
    workspace = WorkspaceBuilder().create(sample_repository, tmp_path / "worktree")

    (workspace.root / "src" / "parser.py").write_text("CHANGED = True\n", encoding="utf-8")

    assert workspace.changed_files() == ["src/parser.py"]
    assert "diff --git" in workspace.patch()
    assert (sample_repository / "src" / "parser.py").read_text(encoding="utf-8") == original


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Docker smoke disabled",
)
@pytest.mark.asyncio
async def test_pinned_sandbox_image_runs_without_network(tmp_path) -> None:
    sandbox = DockerSandbox(DockerSettings(), tmp_path)

    sandbox.check_ready()
    result = await sandbox.run(("python", "-c", "print('sandbox-ready')"))

    assert result.exit_code == 0
    assert result.stdout.strip() == "sandbox-ready"
