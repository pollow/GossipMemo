"""Internal projections and read views over stored records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import (
    CoverageEntryStatus,
    HypothesisConfidence,
    HypothesisEvidence,
    HypothesisOwnerKind,
    HypothesisStatus,
    LearningGoalStatus,
    MemoryKind,
)


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


class ContinuityView(BaseModel):
    text: str = ""
    related_person_ids: list[str] = Field(default_factory=list)
    through_message_id: str | None = None


class OwnerEvidenceDigestView(BaseModel):
    summary: str = Field(min_length=1, max_length=600)
    source_memory_ids: list[str] = Field(min_length=1, max_length=512)
    basis: str = "explicit"
    uncertainty: str = ""
    semantic_subject: str = ""


class CoverageEntryView(BaseModel):
    """One stored summary of how well a path under a root is understood.

    An entry is a summary over many Memories, not a Memory: it says what we
    understand about this path, never what is still missing.  `path` is free
    text and deliberately unnormalized; the root-level entry is the one with
    an empty `path`.
    """

    id: str
    space_id: str
    root: str
    path: str = ""
    content: str
    status: CoverageEntryStatus = "active"
    created_at: str
    updated_at: str


class CoverageRootView(BaseModel):
    """One root's audit cursor: how far its own evidence backlog is read."""

    space_id: str
    root: str
    revision: int = 0
    source_watermark: str | None = None
    source_cursor_id: str | None = None


class LearningGoalView(BaseModel):
    """A stored learning direction.

    `focus_kind`/`focus_id` are derived by storage from the goal's own text
    through deterministic alias matching, never supplied by the model.
    """

    id: str
    space_id: str
    prompt: str
    rationale: str
    entry_ids: list[str] = Field(default_factory=list)
    focus_kind: Literal["user", "person", "relationship"] = "user"
    focus_id: str | None = None
    status: LearningGoalStatus
    status_reason: str | None = None
    created_at: str
    updated_at: str


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


class PersonSummaryView(BaseModel):
    """A lightweight row for listing/searching people -- no profile synthesis.

    Carries aliases (unlike `PersonView`) so a caller can judge whether two
    rows are plausibly the same person, which is the whole point of
    `list_people`: surfacing merge candidates for `merge_person`.
    """

    id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)


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
