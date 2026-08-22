"""The concrete SQLite world store and the `WorldStore` protocol it satisfies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..models import (
    ContinuityReasoningResult,
    ContinuityView,
    CoverageEntryView,
    CoverageRootView,
    ExtractedCoverageAudit,
    ExtractionResult,
    GoalPlanningResult,
    HypothesisView,
    LearningGoalView,
    MemoryView,
    ModelMessage,
    PersonReasoningResult,
    PersonView,
    RelationshipReasoningResult,
    RelationshipView,
    UserModelReasoningResult,
    UserModelView,
)
from ._admin import _AdminReadMixin
from ._context import _ContextMixin
from ._messages import DEFAULT_EXTRACTION_COMPARISON_LIMIT, PendingExtraction


class WorldStore(Protocol):
    """The store surface the reasoners depend on, and nothing else.

    Every method here is called by a module under `gossipmemo.reasoners`,
    and every store call those modules make is declared here. Signatures
    mirror `SqliteWorldStore` exactly. `tests/test_store_protocol.py`
    enforces both directions.
    """

    def pending_extractions(self) -> list[PendingExtraction]: ...

    def load_batch(self, space_id: str, batch_id: str) -> list[ModelMessage]: ...

    def load_extraction_context(
        self, space_id: str, batch_id: str, limit: int = 2
    ) -> list[ModelMessage]: ...

    def load_known_people(
        self, space_id: str, messages: list[ModelMessage]
    ) -> list[dict[str, Any]]: ...

    def load_extraction_comparisons(
        self, space_id: str, batch_id: str,
        limit: int = DEFAULT_EXTRACTION_COMPARISON_LIMIT,
    ) -> list[MemoryView]: ...

    def mark_extraction_attempt(self, space_id: str, batch_id: str) -> None: ...

    def fail_extraction(self, space_id: str, batch_id: str, error: str) -> None: ...

    def apply_extraction(
        self, space_id: str, batch_id: str, result: ExtractionResult,
        comparison_memory_ids: set[str] | None = None,
    ) -> tuple[set[str], set[str]]: ...

    def stale_entities(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]: ...

    def owner_review_context(
        self, space_id: str, owner_kind: str, owner_id: str | None,
    ) -> tuple[list[MemoryView], list[HypothesisView]]: ...

    def user_model_context(
        self, space_id: str, *, delta_only: bool = False
    ) -> tuple[UserModelView, list[MemoryView], str | None] | None: ...

    def apply_user_model_reasoning(
        self, space_id: str, expected_watermark: str | None,
        result: UserModelReasoningResult,
        context_hypothesis_ids: set[str] | None = None,
    ) -> bool: ...

    def person_context(
        self, space_id: str, person_id: str, *, delta_only: bool = False
    ) -> tuple[PersonView, list[MemoryView], str | None] | None: ...

    def apply_person_reasoning(
        self,
        space_id: str,
        person_id: str,
        expected_watermark: str | None,
        result: PersonReasoningResult,
        context_inferred_memory_ids: set[str] | None = None,
        context_hypothesis_ids: set[str] | None = None,
    ) -> bool: ...

    def relationship_context(
        self, space_id: str, relationship_id: str, *, delta_only: bool = False
    ) -> tuple[RelationshipView, list[MemoryView], str | None] | None: ...

    def apply_relationship_reasoning(
        self,
        space_id: str,
        relationship_id: str,
        expected_watermark: str | None,
        result: RelationshipReasoningResult,
        context_inferred_memory_ids: set[str] | None = None,
        context_hypothesis_ids: set[str] | None = None,
    ) -> bool: ...

    def coverage_context(
        self, space_id: str, limit: int | None = 400,
    ) -> tuple[CoverageRootView, list[CoverageEntryView], list[MemoryView]] | None: ...

    def apply_coverage_audit(
        self, space_id: str, root: str, expected_watermark: str | None,
        expected_cursor_id: str | None, audit: ExtractedCoverageAudit,
        chunk: list[MemoryView], context_entry_ids: set[str],
    ) -> bool: ...

    def learning_goal_context(self, space_id: str) -> tuple[
        int, list[CoverageEntryView], list[LearningGoalView], list[LearningGoalView],
    ] | None: ...

    def apply_goal_planning(
        self, space_id: str, expected_revision: int, result: GoalPlanningResult,
        context_goal_ids: set[str],
    ) -> bool: ...

    def continuity_context(
        self, space_id: str
    ) -> tuple[ContinuityView | None, list[ModelMessage]] | None: ...

    def search_vectors(
        self,
        space_id: str,
        owner_kind: str,
        query_vector: Sequence[float],
        k: int,
        *,
        statuses: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]: ...

    def apply_continuity_reasoning(
        self, space_id: str, expected_through_message_id: str | None,
        result: ContinuityReasoningResult,
    ) -> bool: ...


class SqliteWorldStore(_ContextMixin, _AdminReadMixin):
    """SQLite Adapter. Each method owns its short atomic write internally.

    `_AdminReadMixin` is composed in for the admin UI only; it is
    deliberately not part of the `WorldStore` protocol above (see its
    module docstring).
    """


__all__ = [
    "DEFAULT_EXTRACTION_COMPARISON_LIMIT",
    "PendingExtraction",
    "SqliteWorldStore",
    "WorldStore",
]
