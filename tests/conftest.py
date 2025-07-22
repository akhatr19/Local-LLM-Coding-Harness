import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sample_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "src").mkdir()
    (repository / "build").mkdir()
    (repository / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repository / "README.md").write_text("# Parser example\n", encoding="utf-8")
    (repository / "ignored.txt").write_text("do not index\n", encoding="utf-8")
    (repository / "build" / "generated.py").write_text("GENERATED = True\n", encoding="utf-8")
    (repository / "src" / "parser.py").write_text(
        '"""Small parser fixture."""\n'
        "\n"
        "class Parser:\n"
        "    def parse(self, value: str) -> str:\n"
        "        # TODO: normalize input\n"
        "        return value\n"
        "\n"
        "async def parse_async(value: str) -> str:\n"
        "    return value\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@localhost",
            "commit",
            "-m",
            "initial fixture",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return repository
