"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .common import MemoryKind, utc_now
from .views import (
    ContextBundle,
    GuidanceBundle,
    MemoryView,
    PersonView,
    QueryContext,
)


class SourceRef(BaseModel):
    provider: str = "agent_chat"
    conversation_key: str | None = None
    item_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageInput(BaseModel):
    idempotency_key: str | None = None
    author: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    occurred_at: datetime = Field(default_factory=utc_now)
    source: SourceRef = Field(default_factory=SourceRef)

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(timezone.utc)


class TurnRequest(BaseModel):
    """A batch of messages (durable write), plus the SDK's cached context watermark.

    This is the single write path: a batch whose last message is not from
    the user is a plain durable write. See `SocialMemoryWorld.turn` for the
    read-enrichment rule that follows from that.
    """

    messages: list[MessageInput] = Field(min_length=1, max_length=100)
    context_version: str | None = None
    memory_limit: int = Field(default=5, ge=1, le=10)
    # Learning-goal sampling knobs, both defaulting to the historical
    # behavior: a random 3-5 goals seeded from the context version. See
    # `SqliteWorldStore._guidance`.
    goals: int | None = Field(default=None, ge=0, le=200)
    goal_seed: str | None = None


class TurnResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    message_ids: list[str]
    known_people: list[PersonView] = Field(default_factory=list)
    memory_recall: list[MemoryView] = Field(default_factory=list)
    guidance: GuidanceBundle = Field(default_factory=GuidanceBundle)
    context_update: ContextBundle | None = None
    context_status: Literal["available", "unavailable"] = "available"


class MergePersonRequest(BaseModel):
    target_person_id: str = Field(min_length=1)


class MergePersonResponse(BaseModel):
    source_person_id: str
    target_person_id: str
    status: Literal["merged"] = "merged"
    affected_relationship_ids: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    people: list[str] = Field(default_factory=list)
    include_relationships: bool = True
    expand_relationships: Literal[0, 1] = 0
    include_evidence: bool = True
    limit: int = Field(default=30, ge=1, le=100)


class QueryResponse(QueryContext):
    answer: str


class ManualMemoryRequest(BaseModel):
    content: str = Field(min_length=1)
    kind: MemoryKind = "fact"
    people: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    about_user: bool = False


class RetractRequest(BaseModel):
    reason: str | None = None


class SupersedeRequest(BaseModel):
    content: str = Field(min_length=1)
    kind: MemoryKind | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    reason: str | None = None
    about_user: bool | None = None


class QueueStatus(BaseModel):
    pending: int
    running: bool
    current_label: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_configured: bool
    queue: QueueStatus
    embedding_enabled: bool = False
    embedding_pending: int = 0
