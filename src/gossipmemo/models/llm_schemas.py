"""Structured-output schemas the LLM emits, and the schemas it is shown."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import (
    CoverageEntryStatus,
    HypothesisConfidence,
    HypothesisEvidence,
    MemoryBasis,
    MemoryKind,
)


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


class ExtractedCoverageEntry(BaseModel):
    """A new entry under the audited root; storage assigns its trusted ID."""

    path: str = ""
    content: str = Field(min_length=1)


class ExtractedCoverageEntryEdit(BaseModel):
    """A rewrite of one entry listed in the prompt.

    Merge and split need no dedicated operations: merging is rewriting one
    entry and marking the other `superseded`, splitting is narrowing one
    entry and adding another.  Omitted `path` keeps the stored path.
    """

    entry_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    path: str | None = None
    status: CoverageEntryStatus = "active"


class ExtractedCoverageAudit(BaseModel):
    additions: list[ExtractedCoverageEntry] = Field(default_factory=list)
    modifications: list[ExtractedCoverageEntryEdit] = Field(default_factory=list)


class LearningGoalUpsert(BaseModel):
    """One learning direction, carried entirely in natural language.

    `rationale` says which direction this is and why it is worth knowing;
    `prompt` is one suggested wording.  `entry_ids` points back at the
    coverage entries the direction grew out of and is best effort only:
    storage drops IDs it cannot resolve and still stores the goal, so a
    direction that files nowhere is never lost.  Focus is resolved by
    storage from the goal's own text, so the model never picks a person or
    relationship ID.
    """

    goal_id: str | None = None
    prompt: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    entry_ids: list[str] = Field(default_factory=list)


class LearningGoalTransition(BaseModel):
    goal_id: str = Field(min_length=1)
    status: Literal["partial", "answered", "deferred", "retired", "open"]
    reason: str = Field(min_length=1)


class GoalPlanningResult(BaseModel):
    upserts: list[LearningGoalUpsert] = Field(default_factory=list)
    transitions: list[LearningGoalTransition] = Field(default_factory=list)


class LearningGoalCandidate(BaseModel):
    """A non-mutating proposal from one root's planning request."""

    prompt: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    entry_ids: list[str] = Field(default_factory=list)


class GoalClosureRecommendation(BaseModel):
    """A per-root vote that an existing goal now looks settled.

    Non-mutating: it is evidence for reconciliation, the only pass that may
    actually transition a goal, not an instruction it must follow.
    """

    goal_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GoalPlanningCandidates(BaseModel):
    candidates: list[LearningGoalCandidate] = Field(default_factory=list)
    closure_recommendations: list[GoalClosureRecommendation] = Field(default_factory=list)
