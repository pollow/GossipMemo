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
CoverageLevel = Literal["unknown", "fragmentary", "grounded", "rich"]
CoverageBoundaryKind = Literal["edge", "blind_spot", "conflict"]
CoverageBoundaryStatus = Literal["open", "resolved"]
LearningGoalStatus = Literal["open", "partial", "answered", "deferred", "retired"]

# These are intentionally prompt-native IDs rather than a normalized rubric
# table: they are a stable contract for the CoverageAudit and GoalPlanning LLM
# calls, while each space stores only its current rebuildable assessment.
COVERAGE_CRITERIA: dict[str, str] = {
    "M1": "life_chapters", "M2": "everyday_life", "M3": "turning_points",
    "M4": "people_and_relationship_arcs", "M5": "places_and_context",
    "M6": "lived_scenes", "M7": "inner_experience", "M8": "themes_and_change",
    "M9": "unresolved_threads", "P1": "identity_and_self_story",
    "P2": "values_and_tradeoffs", "P3": "worldview_and_beliefs",
    "P4": "goals_motives_fears", "P5": "reasoning_and_decisions",
    "P6": "emotional_patterns", "P7": "social_style_and_boundaries",
    "P8": "preferences_and_routines", "P9": "voice_and_expression",
    "P10": "context_and_exceptions", "P11": "skills_and_knowledge",
}


def coverage_skeleton() -> dict[str, dict[str, str]]:
    return {criterion_id: {"level": "unknown"} for criterion_id in COVERAGE_CRITERIA}


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


class GuidanceItem(BaseModel):
    """Small, tentative prompts that may help an agent guide a conversation."""
    id: str
    kind: Literal["hypothesis", "learning_goal"]
    content: str
    owner_kind: Literal["user", "person", "relationship"]
    owner_id: str | None = None
    status: str
    confidence: HypothesisConfidence | None = None


class GuidanceBundle(BaseModel):
    items: list[GuidanceItem] = Field(default_factory=list)


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

class ExtractedOwnerEvidenceDigestItem(BaseModel):
    summary: str = Field(min_length=1, max_length=600)
    source_memory_ids: list[str] = Field(min_length=1, max_length=32)
    basis: str = "explicit"
    uncertainty: str = ""
    semantic_subject: str = ""

class ExtractedOwnerEvidenceDigest(BaseModel):
    items: list[ExtractedOwnerEvidenceDigestItem] = Field(max_length=16)

class OwnerEvidenceDigestView(BaseModel):
    summary: str = Field(min_length=1, max_length=600)
    source_memory_ids: list[str] = Field(min_length=1, max_length=512)
    basis: str = "explicit"
    uncertainty: str = ""
    semantic_subject: str = ""


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


class CoverageCriterionPatch(BaseModel):
    criterion_id: str
    level: CoverageLevel
    known_state: str = ""
    evidence_memory_ids: list[str] = Field(default_factory=list)


class CoverageBoundary(BaseModel):
    id: str = Field(min_length=1)
    kind: CoverageBoundaryKind
    status: CoverageBoundaryStatus = "open"
    summary: str = Field(min_length=1)
    criterion_refs: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str | None = None
    status_reason: str | None = None


class CoverageBoundaryUpsert(BaseModel):
    """A new boundary only; storage assigns its trusted ID."""

    kind: CoverageBoundaryKind
    summary: str = Field(min_length=1)
    criterion_refs: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str | None = None


class CoverageBoundaryTransition(BaseModel):
    boundary_id: str = Field(min_length=1)
    status: CoverageBoundaryStatus
    reason: str = Field(min_length=1)


class CoverageAuditPatch(BaseModel):
    criteria: list[CoverageCriterionPatch] = Field(default_factory=list)
    boundary_upserts: list[CoverageBoundaryUpsert] = Field(default_factory=list)
    boundary_transitions: list[CoverageBoundaryTransition] = Field(default_factory=list)
    life_periods: list[str] = Field(default_factory=list)
    relationship_arcs: list[str] = Field(default_factory=list)
    behavioral_contexts: list[str] = Field(default_factory=list)


class CoverageMapView(BaseModel):
    space_id: str
    revision: int = 0
    source_watermark: str | None = None
    source_cursor_id: str | None = None
    criteria: dict[str, dict[str, Any]] = Field(default_factory=coverage_skeleton)
    boundaries: list[CoverageBoundary] = Field(default_factory=list)
    life_periods: list[str] = Field(default_factory=list)
    relationship_arcs: list[str] = Field(default_factory=list)
    behavioral_contexts: list[str] = Field(default_factory=list)


class LearningGoalUpsert(BaseModel):
    goal_id: str | None = None
    prompt: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    criteria_refs: list[str] = Field(min_length=1)
    boundary_ids: list[str] = Field(min_length=1)
    focus_kind: Literal["user", "person", "relationship"] = "user"
    focus_id: str | None = None


class LearningGoalTransition(BaseModel):
    goal_id: str = Field(min_length=1)
    status: Literal["partial", "answered", "deferred", "retired", "open"]
    reason: str = Field(min_length=1)


class GoalPlanningResult(BaseModel):
    upserts: list[LearningGoalUpsert] = Field(default_factory=list)
    transitions: list[LearningGoalTransition] = Field(default_factory=list)


class LearningGoalCandidate(BaseModel):
    """A non-mutating proposal used while planning cannot fit in one request."""

    prompt: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    criteria_refs: list[str] = Field(min_length=1)
    boundary_ids: list[str] = Field(min_length=1)
    focus_kind: Literal["user", "person", "relationship"] = "user"
    focus_id: str | None = None


class GoalPlanningCandidates(BaseModel):
    candidates: list[LearningGoalCandidate] = Field(default_factory=list)


class LearningGoalView(BaseModel):
    id: str
    space_id: str
    prompt: str
    rationale: str
    criteria_refs: list[str] = Field(default_factory=list)
    boundary_ids: list[str] = Field(default_factory=list)
    focus_kind: Literal["user", "person", "relationship"] = "user"
    focus_id: str | None = None
    status: LearningGoalStatus
    status_reason: str | None = None
    created_at: str
    updated_at: str


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
    guidance: GuidanceBundle = Field(default_factory=GuidanceBundle)


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
