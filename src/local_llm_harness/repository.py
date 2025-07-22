"""Safe, read-only repository inspection tools."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".harness",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


class RepositoryError(RuntimeError):
    """Base repository inspection error."""


class UnsafeRepositoryPathError(RepositoryError):
    """A requested path escaped the repository root."""


class BinaryFileError(RepositoryError):
    """A text operation was requested for a binary file."""


class FileTooLargeError(RepositoryError):
    """A file exceeded the configured inspection limit."""


class RepositoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileSlice(RepositoryModel):
    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str


class SearchMatch(RepositoryModel):
    path: str
    line: int
    column: int
    text: str


class SymbolInfo(RepositoryModel):
    path: str
    name: str
    kind: str
    start_line: int
    end_line: int


class GitCommit(RepositoryModel):
    commit: str
    author: str
    authored_at: str
    subject: str


class RepositoryInspector:
    """Expose bounded repository operations without accepting shell command strings."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = 1_000_000,
        max_read_lines: int = 500,
        command_timeout_seconds: float = 30,
    ) -> None:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise RepositoryError(f"repository directory does not exist: {root}")
        self.root = resolved
        self.max_file_bytes = max_file_bytes
        self.max_read_lines = max_read_lines
        self.command_timeout_seconds = command_timeout_seconds

    @property
    def is_git_repository(self) -> bool:
        result = self._run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def list_files(self) -> list[str]:
        """List tracked and untracked, non-ignored files in deterministic order."""

        if self.is_git_repository:
            result = self._run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
            )
            candidates = [path for path in result.stdout.split("\0") if path]
        else:
            candidates = [
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob("*")
                if path.is_file()
            ]
        return sorted(
            path
            for path in candidates
            if not any(part in EXCLUDED_DIRECTORIES for part in PurePosixPath(path).parts)
        )

    def read_file(
        self,
        relative_path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> FileSlice:
        path = self._resolve_file(relative_path)
        if start_line < 1:
            raise RepositoryError("start_line must be at least 1")
        if end_line is not None and end_line < start_line:
            raise RepositoryError("end_line must not precede start_line")
        if end_line is not None and end_line - start_line + 1 > self.max_read_lines:
            raise RepositoryError(f"requested range exceeds {self.max_read_lines} lines")

        raw = self._read_bytes(path)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryFileError(f"file is not UTF-8 text: {relative_path}") from exc
        lines = text.splitlines(keepends=True)
        requested_end = end_line or min(len(lines), start_line + self.max_read_lines - 1)
        actual_end = min(requested_end, len(lines))
        content = "".join(lines[start_line - 1 : actual_end])
        return FileSlice(
            path=PurePosixPath(relative_path).as_posix(),
            start_line=start_line,
            end_line=max(start_line - 1, actual_end),
            total_lines=len(lines),
            content=content,
        )

    def lexical_search(
        self,
        query: str,
        *,
        limit: int = 30,
        regex: bool = False,
    ) -> list[SearchMatch]:
        if not query:
            raise RepositoryError("search query cannot be empty")
        if limit < 1:
            raise RepositoryError("search limit must be at least 1")
        if shutil.which("rg") is None:
            raise RepositoryError("ripgrep (rg) is required for lexical search")

        command = ["rg", "--json", "--line-number", "--hidden", "--color", "never"]
        if not regex:
            command.append("--fixed-strings")
        for directory in sorted(EXCLUDED_DIRECTORIES):
            command.extend(["--glob", f"!{directory}/**"])
        command.extend([query, "."])
        result = self._run(command, check=False)
        if result.returncode not in {0, 1}:
            raise RepositoryError(f"ripgrep failed: {result.stderr.strip()}")

        matches: list[SearchMatch] = []
        for line in result.stdout.splitlines():
            event = json.loads(line)
            if event.get("type") != "match":
                continue
            data = event["data"]
            submatches = data.get("submatches", [])
            column = int(submatches[0]["start"]) + 1 if submatches else 1
            path = PurePosixPath(data["path"]["text"]).as_posix()
            if path.startswith("./"):
                path = path[2:]
            matches.append(
                SearchMatch(
                    path=path,
                    line=int(data["line_number"]),
                    column=column,
                    text=data["lines"]["text"].rstrip("\r\n"),
                )
            )
            if len(matches) >= limit:
                break
        return matches

    def python_symbols(self, relative_path: str) -> list[SymbolInfo]:
        file_slice = self.read_file(relative_path, end_line=self.max_read_lines)
        if file_slice.total_lines > self.max_read_lines:
            path = self._resolve_file(relative_path)
            raw = self._read_bytes(path)
            try:
                source = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise BinaryFileError(f"file is not UTF-8 text: {relative_path}") from exc
        else:
            source = file_slice.content
        try:
            tree = ast.parse(source, filename=relative_path)
        except SyntaxError as exc:
            raise RepositoryError(f"cannot parse Python file {relative_path}: {exc}") from exc

        symbols = []
        node_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        for node in ast.walk(tree):
            if not isinstance(node, node_types):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if isinstance(node, ast.AsyncFunctionDef):
                kind = "async_function"
            symbols.append(
                SymbolInfo(
                    path=relative_path,
                    name=node.name,
                    kind=kind,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                )
            )
        return sorted(symbols, key=lambda symbol: (symbol.start_line, symbol.name))

    def git_status(self) -> str:
        self._require_git()
        return self._run(["git", "status", "--short"]).stdout

    def git_history(self, *, limit: int = 20, relative_path: str | None = None) -> list[GitCommit]:
        self._require_git()
        if limit < 1 or limit > 200:
            raise RepositoryError("git history limit must be between 1 and 200")
        command = [
            "git",
            "log",
            f"-n{limit}",
            "--format=%H%x1f%an%x1f%aI%x1f%s",
        ]
        if relative_path is not None:
            self._resolve_file(relative_path)
            command.extend(["--", relative_path])
        result = self._run(command)
        commits = []
        for line in result.stdout.splitlines():
            commit, author, authored_at, subject = line.split("\x1f", 3)
            commits.append(
                GitCommit(
                    commit=commit,
                    author=author,
                    authored_at=authored_at,
                    subject=subject,
                )
            )
        return commits

    def git_diff(self, *, ref: str | None = None, relative_path: str | None = None) -> str:
        self._require_git()
        command = ["git", "diff", "--no-ext-diff", "--unified=3"]
        if ref is not None:
            if ref.startswith("-") or any(character.isspace() for character in ref):
                raise RepositoryError("invalid Git reference")
            command.append(ref)
        if relative_path is not None:
            self._resolve_file(relative_path)
            command.extend(["--", relative_path])
        return self._run(command).stdout

    def current_commit(self) -> str | None:
        if not self.is_git_repository:
            return None
        result = self._run(["git", "rev-parse", "HEAD"], check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def file_bytes(self, relative_path: str) -> bytes:
        """Return bounded file bytes for indexing and fingerprinting."""

        return self._read_bytes(self._resolve_file(relative_path))

    def _resolve_file(self, relative_path: str) -> Path:
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise UnsafeRepositoryPathError("path must be repository-relative")
        resolved = self.root.joinpath(*pure_path.parts).resolve()
        if not resolved.is_relative_to(self.root):
            raise UnsafeRepositoryPathError("path escapes repository root")
        if not resolved.is_file():
            raise RepositoryError(f"repository file does not exist: {relative_path}")
        return resolved

    def _read_bytes(self, path: Path) -> bytes:
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise FileTooLargeError(
                f"file exceeds {self.max_file_bytes} byte inspection limit: {path.name}"
            )
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            raise BinaryFileError(f"binary file cannot be inspected as text: {path.name}")
        return raw

    def _require_git(self) -> None:
        if not self.is_git_repository:
            raise RepositoryError(f"not a Git repository: {self.root}")

    def _run(self, command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RepositoryError(f"repository command failed: {command[0]}: {exc}") from exc
        if check and result.returncode != 0:
            raise RepositoryError(
                f"repository command failed ({result.returncode}): {result.stderr.strip()}"
            )
        return result
