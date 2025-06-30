"""Local LLM Coding Harness."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-llm-coding-harness")
except PackageNotFoundError:  # pragma: no cover - only used from an unpackaged source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
