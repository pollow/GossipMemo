from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


MemoryBasis = Literal["stated", "observed", "reported", "inferred", "manual"]
MemoryKind = Literal["fact", "event", "preference", "plan", "situation", "impression"]
HypothesisOwnerKind = Literal["user", "person", "relationship"]
HypothesisStatus = Literal["open", "promoted", "rejected", "superseded", "retired"]
HypothesisConfidence = Literal["low", "medium", "high"]
HypothesisEvidenceRole = Literal["support", "counter"]


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


class IngestRequest(BaseModel):
    messages: list[MessageInput] = Field(min_length=1, max_length=100)


class IngestResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    message_ids: list[str]


class TurnRequest(BaseModel):
    """One user turn, plus the SDK's cached context watermark."""

    message: MessageInput
    context_version: str | None = None
    memory_limit: int = Field(default=5, ge=1, le=10)

    @field_validator("message")
    @classmethod
    def require_user_message(cls, value: MessageInput) -> MessageInput:
        if value.author != "user":
            raise ValueError("turn message author must be user")
        return value


class TurnResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    message_id: str
    known_people: list[PersonView] = Field(default_factory=list)
    memory_recall: list[MemoryView] = Field(default_factory=list)
    context_update: ContextBundle | None = None
    context_status: Literal["available", "unavailable"] = "available"


class MergePersonRequest(BaseModel):
    target_person_id: str = Field(min_length=1)


class MergePersonResponse(BaseModel):
    source_person_id: str
    target_person_id: str
    status: Literal["merged"] = "merged"
    affected_relationship_ids: list[str] = Field(default_factory=list)


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
    supersedes_memory_id: str | None = None


class ExtractionResult(BaseModel):
    people: list[ExtractedPerson] = Field(default_factory=list)
    memories: list[ExtractedMemory] = Field(default_factory=list)


class InferredMemory(BaseModel):
    content: str
    kind: MemoryKind = "impression"
    source_memory_ids: list[str] = Field(min_length=1)


class InferredMemoryRetraction(BaseModel):
    """An explicit request to retract one target-owned inferred Memory."""

    memory_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class InferredMemoryActions(BaseModel):
    """Lifecycle mutations emitted for inferred Memories.

    Omitting an item is deliberately not a retraction.  A caller must provide
    both the inferred Memory IDs it showed to the reasoner and an explicit,
    reasoned retraction action before an existing inference can be retracted.
    """

    upserts: list[InferredMemory] = Field(default_factory=list)
    retractions: list[InferredMemoryRetraction] = Field(default_factory=list)


class HypothesisEvidence(BaseModel):
    memory_id: str = Field(min_length=1)
    role: HypothesisEvidenceRole = "support"


class HypothesisUpsert(BaseModel):
    """A tentative claim independent of any inferred Memory."""

    hypothesis_id: str | None = None
    content: str = Field(min_length=1)
    kind: MemoryKind = "impression"
    confidence: HypothesisConfidence = "low"
    evidence: list[HypothesisEvidence] = Field(min_length=1)


class HypothesisTransition(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    status: Literal["promoted", "rejected", "superseded", "retired"]
    reason: str = Field(min_length=1)
    promoted_memory_id: str | None = None


class HypothesisActions(BaseModel):
    """Additive hypothesis upserts plus explicitly scoped transitions."""

    upserts: list[HypothesisUpsert] = Field(default_factory=list)
    transitions: list[HypothesisTransition] = Field(default_factory=list)


class HypothesisView(BaseModel):
    """Durable tentative claim and its non-inferred evidence."""

    id: str
    space_id: str
    owner_kind: HypothesisOwnerKind
    owner_id: str | None = None
    content: str
    kind: MemoryKind
    confidence: HypothesisConfidence
    status: HypothesisStatus
    promoted_memory_id: str | None = None
    evidence: list[HypothesisEvidence] = Field(default_factory=list)
    created_at: str
    updated_at: str


class PersonReasoningResult(BaseModel):
    profile_card: dict[str, Any] = Field(default_factory=dict)
    inferred_memories: list[InferredMemory] = Field(default_factory=list)
    inferred_memory_actions: InferredMemoryActions | None = None
    hypothesis_actions: HypothesisActions | None = None


class UserModelReasoningResult(BaseModel):
    """Bounded, rebuildable profile projection for the current user."""

    profile_card: dict[str, Any] = Field(default_factory=dict)
    hypothesis_actions: HypothesisActions | None = None


class ContinuityReasoningResult(BaseModel):
    text: str = ""
    related_person_ids: list[str] = Field(default_factory=list)
    through_message_id: str


class ContinuityView(BaseModel):
    text: str = ""
    related_person_ids: list[str] = Field(default_factory=list)
    through_message_id: str | None = None


class RelationshipReasoningResult(BaseModel):
    facets: list[dict[str, Any]] = Field(default_factory=list)
    closeness: str | None = None
    tone: str | None = None
    status: str = "unknown"
    summary: str = ""
    inferred_memories: list[InferredMemory] = Field(default_factory=list)
    inferred_memory_actions: InferredMemoryActions | None = None
    hypothesis_actions: HypothesisActions | None = None


class PersonProjectionResult(BaseModel):
    profile_card: dict[str, Any] = Field(default_factory=dict)


class RelationshipProjectionResult(BaseModel):
    facets: list[dict[str, Any]] = Field(default_factory=list)
    closeness: str | None = None
    tone: str | None = None
    status: str = "unknown"
    summary: str = ""


class ReasoningActionsResult(BaseModel):
    inferred_memory_actions: InferredMemoryActions | None = None
    hypothesis_actions: HypothesisActions | None = None


class UserReasoningActionsResult(BaseModel):
    """User review never creates inferred Memories directly."""

    hypothesis_actions: HypothesisActions | None = None


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
    valid_from: str | None = None
    valid_to: str | None = None


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


class ContextBundle(BaseModel):
    version: str
    user_model: UserModelView | None = None
    continuity: ContinuityView | None = None
    people: list[PersonView] = Field(default_factory=list)


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
