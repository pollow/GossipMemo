"""Relationship projection reasoner."""

from __future__ import annotations

from ..llm import LlmModel
from ..queue import ReasonerCallQueue
from ..store import WorldStore


class RelationshipReasoner:
    """Refresh one stale Relationship projection per attempt.

    Mirrors `PersonReasoner`: an in-flight watermark conflict simply causes
    the next `attempt` to recompute from the latest snapshot.
    """

    name = "relationship"

    def __init__(self, store: WorldStore, model: LlmModel, queue: ReasonerCallQueue) -> None:
        self.store = store
        self.model = model
        self.queue = queue

    async def attempt(self, space_id: str) -> bool:
        _, relationships, _ = self.store.stale_entities()
        targets = [relationship_id for sid, relationship_id in relationships if sid == space_id]
        if not targets:
            return False
        relationship_id = targets[0]
        more_targets = len(targets) > 1
        context = self.store.relationship_context(space_id, relationship_id)
        if not context:
            return more_targets
        relationship, memories, watermark = context
        if not relationship.stale:
            return more_targets
        inferred, hypotheses = self.store.owner_review_context(space_id, "relationship", relationship_id)
        result = await self.queue.submit(
            "reason-relationship",
            self.model.reason_relationship,
            relationship, memories, inferred, hypotheses,
        )
        applied = self.store.apply_relationship_reasoning(
            space_id,
            relationship_id,
            watermark,
            result,
            {memory.id for memory in inferred},
            {hypothesis.id for hypothesis in hypotheses},
        )
        return more_targets or not applied


__all__ = ["RelationshipReasoner"]
