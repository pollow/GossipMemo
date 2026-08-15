"""Small, dependency-free structured logging setup for the server."""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """Render event messages and structured fields as JSON or readable text."""

    _standard = set(logging.LogRecord(None, 0, "", 0, "", (), None).__dict__)
    _sensitive = (
        "api_key",
        "apikey",
        "authorization",
        "token",
        "password",
        "secret",
        "body",
        "content",
        "prompt",
    )

    def __init__(self, mode: str = "json") -> None:
        super().__init__()
        self.mode = mode

    def format(self, record: logging.LogRecord) -> str:
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._standard
            and not key.startswith("_")
            and not any(word in key.lower() for word in self._sensitive)
        }
        request_id = request_id_context.get()
        if request_id and "request_id" not in fields:
            fields["request_id"] = request_id
        if self.mode == "text":
            suffix = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
            return f"{record.levelname} {record.name} {record.getMessage()}" + (f" {suffix}" if suffix else "")
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", format: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    handler = next(
        (
            candidate
            for candidate in root.handlers
            if getattr(candidate, "_gossipmemo", False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler._gossipmemo = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(StructuredFormatter(format))


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


__all__ = ["StructuredFormatter", "configure_logging", "elapsed_ms", "request_id_context"]
