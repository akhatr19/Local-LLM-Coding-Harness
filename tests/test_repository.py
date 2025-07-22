from pathlib import Path

import pytest

from local_llm_harness.repository import (
    BinaryFileError,
    FileTooLargeError,
    RepositoryError,
    RepositoryInspector,
    UnsafeRepositoryPathError,
)


def test_git_file_listing_respects_ignore_and_generated_directories(sample_repository) -> None:
    inspector = RepositoryInspector(sample_repository)

    files = inspector.list_files()

    assert "src/parser.py" in files
    assert "README.md" in files
    assert "ignored.txt" not in files
    assert "build/generated.py" not in files


def test_bounded_file_read_and_path_protection(sample_repository) -> None:
    inspector = RepositoryInspector(sample_repository, max_read_lines=3)

    content = inspector.read_file("src/parser.py", start_line=3, end_line=5)

    assert content.start_line == 3
    assert content.end_line == 5
    assert "class Parser" in content.content
    with pytest.raises(UnsafeRepositoryPathError):
        inspector.read_file("../outside.txt")
    with pytest.raises(RepositoryError, match="exceeds 3 lines"):
        inspector.read_file("src/parser.py", start_line=1, end_line=4)


def test_binary_and_oversized_files_are_rejected(sample_repository) -> None:
    (sample_repository / "binary.dat").write_bytes(b"hello\0world")
    (sample_repository / "large.txt").write_text("too large", encoding="utf-8")

    with pytest.raises(BinaryFileError):
        RepositoryInspector(sample_repository).read_file("binary.dat")
    with pytest.raises(FileTooLargeError):
        RepositoryInspector(sample_repository, max_file_bytes=4).read_file("large.txt")


def test_lexical_search_returns_structured_matches(sample_repository) -> None:
    inspector = RepositoryInspector(sample_repository)

    matches = inspector.lexical_search("TODO", limit=5)

    assert len(matches) == 1
    assert matches[0].path == "src/parser.py"
    assert matches[0].line == 5
    assert matches[0].column > 1


def test_python_symbol_lookup(sample_repository) -> None:
    symbols = RepositoryInspector(sample_repository).python_symbols("src/parser.py")

    assert [(symbol.name, symbol.kind) for symbol in symbols] == [
        ("Parser", "class"),
        ("parse", "function"),
        ("parse_async", "async_function"),
    ]


def test_git_status_history_and_diff_are_bounded(sample_repository) -> None:
    inspector = RepositoryInspector(sample_repository)
    parser = sample_repository / "src" / "parser.py"
    parser.write_text(parser.read_text(encoding="utf-8") + "\nCHANGED = True\n", encoding="utf-8")

    assert "src/parser.py" in inspector.git_status()
    assert inspector.git_history(limit=1)[0].subject == "initial fixture"
    assert "+CHANGED = True" in inspector.git_diff(relative_path="src/parser.py")
    with pytest.raises(RepositoryError, match="invalid Git reference"):
        inspector.git_diff(ref="--output=/tmp/unsafe")


def test_non_git_directory_still_supports_safe_file_listing(tmp_path: Path) -> None:
    (tmp_path / "file.py").write_text("VALUE = 1\n", encoding="utf-8")

    inspector = RepositoryInspector(tmp_path)

    assert inspector.is_git_repository is False
    assert inspector.list_files() == ["file.py"]
    with pytest.raises(RepositoryError, match="not a Git repository"):
        inspector.git_status()
