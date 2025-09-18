"""Small JSON logging setup for local runs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath

from local_llm_harness.config import LoggingSettings

_STANDARD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(settings: LoggingSettings, artifact_root: Path) -> Path:
    relative = PurePosixPath(settings.json_file)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("logging.json_file must remain inside the artifact directory")
    root = artifact_root.expanduser().resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("logging.json_file must remain inside the artifact directory")
    target.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("local_llm_harness")
    logger.setLevel(settings.level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = JsonFormatter()
    file_handler = RotatingFileHandler(
        target,
        maxBytes=settings.max_bytes,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if settings.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return target
