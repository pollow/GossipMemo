"""Relationship projection reasoner and its prompts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient
from ..models import (
    HypothesisView,
    MemoryView,
    ReasoningActionsResult,
    RelationshipProjectionResult,
    RelationshipReasoningResult,
    RelationshipView,
)
from ..store import WorldStore
from .base import DescriptorReasoner
from .owner import owner_reasoning
from .settings import ReasoningSettings

if TYPE_CHECKING:
    from ..llm import LlmTransport


@dataclass(slots=True)
class _RelationshipTarget:
    """One candidate Relationship, plus whether other stale targets remain.

    `relationship` is `None` when the chosen target vanished between the
    stale scan and the context read, or is no longer stale by the time it
    is read -- either way, `call` skips the model call for this attempt.
    """

    relationship_id: str
    more_targets: bool
    relationship: RelationshipView | None = None
    memories: tuple[MemoryView, ...] = ()
    watermark: str | None = None
    inferred: tuple[MemoryView, ...] = field(default_factory=tuple)
    hypotheses: tuple[HypothesisView, ...] = field(default_factory=tuple)


async def _reason_relationship(
    transport: LlmTransport, settings: ReasoningSettings, relationship: RelationshipView,
    memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    *,
    store: WorldStore | None = None,
    space_id: str | None = None,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> RelationshipReasoningResult:
    projection, actions = await owner_reasoning(
        transport, settings, settings.prompts.relationship_reasoning_system, relationship, memories,
        inferred_memories, hypotheses, RelationshipProjectionResult, ReasoningActionsResult,
        store=store, space_id=space_id, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )
    return RelationshipReasoningResult(
        **projection.model_dump(), **actions.model_dump(exclude_none=True),
    )


def build_relationship_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> DescriptorReasoner:
    """Refresh one stale Relationship projection per attempt.

    Mirrors the person reasoner: an in-flight watermark conflict simply
    causes the next `attempt` to recompute from the latest snapshot.
    """

    reason_relationship = partial(
        _reason_relationship, model, settings,
        store=store, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )

    def load_context(space_id: str) -> _RelationshipTarget | None:
        _, relationships, _ = store.stale_entities()
        targets = [relationship_id for sid, relationship_id in relationships if sid == space_id]
        if not targets:
            return None
        relationship_id = targets[0]
        more_targets = len(targets) > 1
        context = store.relationship_context(space_id, relationship_id)
        if not context:
            return _RelationshipTarget(relationship_id, more_targets)
        relationship, memories, watermark = context
        if not relationship.stale:
            return _RelationshipTarget(relationship_id, more_targets)
        inferred, hypotheses = store.owner_review_context(space_id, "relationship", relationship_id)
        return _RelationshipTarget(
            relationship_id, more_targets, relationship, tuple(memories), watermark,
            tuple(inferred), tuple(hypotheses),
        )

    def call(space_id: str, context: _RelationshipTarget):
        if context.relationship is None:
            return None
        return (
            "reason-relationship",
            partial(reason_relationship, space_id=space_id),
            (context.relationship, context.memories, context.inferred, context.hypotheses),
        )

    def apply(space_id: str, context: _RelationshipTarget, result) -> bool:
        return store.apply_relationship_reasoning(
            space_id,
            context.relationship_id,
            context.watermark,
            result,
            {memory.id for memory in context.inferred},
            {hypothesis.id for hypothesis in context.hypotheses},
        )

    def continue_when(context: _RelationshipTarget, result, applied: bool) -> bool:
        return context.more_targets or not applied

    return DescriptorReasoner("relationship", load_context, call, apply, continue_when)


__all__ = ["build_relationship_reasoner"]
