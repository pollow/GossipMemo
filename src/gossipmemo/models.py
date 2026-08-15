from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


MemoryBasis = Literal["stated", "observed", "reported", "inferred", "manual"]
MemoryKind = Literal["fact", "event", "preference", "plan", "situation", "impression"]
ExtractionPolicy = Literal["conservative", "balanced", "comprehensive"]


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
    extraction_policy: ExtractionPolicy = "balanced"

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(timezone.utc)


class IngestRequest(BaseModel):
    messages: list[MessageInput] = Field(min_length=1, max_length=100)


class IngestResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    message_ids: list[str]


class ExtractedPerson(BaseModel):
    ref: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    person_a_ref: str
    person_b_ref: str
    facets: list[dict[str, Any]] = Field(default_factory=list)


class ExtractedMemory(BaseModel):
    content: str = Field(min_length=1)
    kind: MemoryKind = "fact"
    basis: MemoryBasis
    people: list[str] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    about_user: bool = False


class ExtractionResult(BaseModel):
    people: list[ExtractedPerson] = Field(default_factory=list)
    memories: list[ExtractedMemory] = Field(default_factory=list)


class InferredMemory(BaseModel):
    content: str
    kind: MemoryKind = "impression"
    source_memory_ids: list[str] = Field(min_length=1)


class PersonReasoningResult(BaseModel):
    profile_card: dict[str, Any] = Field(default_factory=dict)
    inferred_memories: list[InferredMemory] = Field(default_factory=list)


class UserModelReasoningResult(BaseModel):
    """Bounded, rebuildable profile projection for the current user."""

    profile_card: dict[str, Any] = Field(default_factory=dict)


class RelationshipReasoningResult(BaseModel):
    facets: list[dict[str, Any]] = Field(default_factory=list)
    closeness: str | None = None
    tone: str | None = None
    status: str = "unknown"
    summary: str = ""
    inferred_memories: list[InferredMemory] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    people: list[str] = Field(default_factory=list)
    include_relationships: bool = True
    expand_relationships: Literal[0, 1] = 0
    include_evidence: bool = True
    limit: int = Field(default=30, ge=1, le=100)


class MemoryView(BaseModel):
    id: str
    content: str
    kind: str
    basis: str
    status: str
    people: list[dict[str, str]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str
    about_user: bool = False


class PersonView(BaseModel):
    id: str
    display_name: str
    profile_card: dict[str, Any] = Field(default_factory=dict)
    profile_source_updated_at: str | None = None
    profile_updated_at: str | None = None
    stale: bool = False


class UserModelView(BaseModel):
    space_id: str
    profile_card: dict[str, Any] = Field(default_factory=dict)
    profile_source_updated_at: str | None = None
    profile_updated_at: str | None = None
    stale: bool = False


class RelationshipView(BaseModel):
    id: str
    person_a_id: str
    person_b_id: str
    facets: list[dict[str, Any]] = Field(default_factory=list)
    closeness: str | None = None
    tone: str | None = None
    status: str
    summary: str
    profile_source_updated_at: str | None = None
    profile_updated_at: str | None = None
    stale: bool = False


class QueryContext(BaseModel):
    people: list[PersonView] = Field(default_factory=list)
    relationships: list[RelationshipView] = Field(default_factory=list)
    memories: list[MemoryView] = Field(default_factory=list)


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


class ModelMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    space_id: str
    author: Literal["user", "assistant"]
    content: str
    occurred_at: str
    source_provider: str
    source_conversation_key: str | None = None
    source_item_id: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_policy: ExtractionPolicy = "balanced"
