"""Normalize portable chat exports into GossipMemo messages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import MessageInput, SourceRef


def load_chat_messages(path: Path) -> list[MessageInput]:
    """Read JSON or JSONL chat records with explicit sender timestamps."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read chat file {path}: {error}") from error
    records = _records(path, raw)
    conversation_default = str(path.resolve())
    messages = [
        _message(path, index, item, conversation_default)
        for index, item in enumerate(records, start=1)
    ]
    return messages


def _records(path: Path, raw: str) -> list[Any]:
    if path.suffix.casefold() == ".jsonl":
        return _jsonl_records(path, raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _jsonl_records(path, raw)
    if isinstance(parsed, dict):
        if "messages" not in parsed:
            raise ValueError(f"chat file {path} JSON object must contain messages")
        parsed = parsed["messages"]
    if not isinstance(parsed, list):
        raise ValueError(
            f"chat file {path} must contain a JSON array or messages object"
        )
    return parsed


def _jsonl_records(path: Path, raw: str) -> list[Any]:
    records: list[Any] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"chat file {path} line {line_number} is not valid JSON"
            ) from error
    return records


def _message(
    path: Path,
    record_number: int,
    item: Any,
    conversation_default: str,
) -> MessageInput:
    label = f"chat file {path} record {record_number}"
    if not isinstance(item, dict):
        raise ValueError(f"{label} must be an object")
    author = item.get("author", item.get("role"))
    if author not in {"user", "assistant"}:
        raise ValueError(f"{label} author/role must be user or assistant")
    content = item.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError(f"{label} content must be non-empty text")
    if "occurred_at" not in item:
        raise ValueError(f"{label} occurred_at is required")
    try:
        occurred_at = datetime.fromisoformat(
            str(item["occurred_at"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(f"{label} occurred_at is invalid") from error

    supplied_source = item.get("source")
    if supplied_source is not None and not isinstance(supplied_source, dict):
        raise ValueError(f"{label} source must be an object")
    source_data = supplied_source or {}
    source_metadata = source_data.get("metadata", {})
    if not isinstance(source_metadata, dict):
        raise ValueError(f"{label} source.metadata must be an object")
    provider = source_data.get("provider", "import")
    conversation_key = source_data.get("conversation_key") or conversation_default
    item_id = source_data.get("item_id")
    if item_id is None:
        item_id = str(record_number - 1)
    metadata = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "author",
            "role",
            "content",
            "occurred_at",
            "source",
            "idempotency_key",
        }
    }
    metadata.update(source_metadata)
    try:
        return MessageInput(
            author=author,
            content=content,
            occurred_at=occurred_at,
            idempotency_key=item.get("idempotency_key")
            or f"import:{provider}:{conversation_key}:{item_id}",
            source=SourceRef(
                provider=provider,
                conversation_key=conversation_key,
                item_id=str(item_id),
                metadata=metadata,
            ),
        )
    except ValidationError as error:
        raise ValueError(f"{label} is invalid: {error.errors()[0]['msg']}") from error


__all__ = ["load_chat_messages"]
