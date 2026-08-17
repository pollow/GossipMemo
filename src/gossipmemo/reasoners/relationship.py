"""Relationship projection reasoner and its prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from ..models import (
    HypothesisView,
    MemoryView,
    RelationshipProjectionResult,
    RelationshipReasoningResult,
    RelationshipView,
)
from ..store import WorldStore
from .base import DescriptorReasoner
from .owner import owner_reasoning

if TYPE_CHECKING:
    from ..llm import LlmTransport

RELATIONSHIP_REASONING_SYSTEM_PROMPT = """Reason carefully about the relationship between
the two endpoint People in the supplied owner context. Linked memories indicate relevance to the
endpoints, not relationship evidence by themselves. Do not transfer the current
user's or either endpoint's standalone traits, preferences, intentions, or actions
into a relationship claim. Mere co-occurrence is not relationship evidence. Look
for recurring interaction patterns, cooperation, friction, trust, initiative, and
meaningful changes in closeness, tone, or status. Recurring patterns require
multiple distinct source memories; a narrow inference from one highly diagnostic
interaction is allowed when calibrated to that evidence. Do not use projections,
inferred memories, or hypotheses as evidence. Distinguish current from historical
conditions. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""


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
    watermark: object = None
    inferred: tuple[MemoryView, ...] = field(default_factory=tuple)
    hypotheses: tuple[HypothesisView, ...] = field(default_factory=tuple)


async def _reason_relationship(
    transport: "LlmTransport", relationship: RelationshipView, memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
) -> RelationshipReasoningResult:
    projection, actions = await owner_reasoning(
        transport, RELATIONSHIP_REASONING_SYSTEM_PROMPT, relationship, memories,
        inferred_memories, hypotheses, RelationshipProjectionResult,
    )
    return RelationshipReasoningResult(
        **projection.model_dump(), **actions.model_dump(exclude_none=True),
    )


def build_relationship_reasoner(store: WorldStore, model: "LlmTransport") -> DescriptorReasoner:
    """Refresh one stale Relationship projection per attempt.

    Mirrors the person reasoner: an in-flight watermark conflict simply
    causes the next `attempt` to recompute from the latest snapshot.
    """

    reason_relationship = partial(_reason_relationship, model)

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
            reason_relationship,
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


__all__ = [
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "build_relationship_reasoner",
]
