"""Relationship projection reasoner and its prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import MemoryView, RelationshipView
from ..prompts import _json
from ..store import WorldStore

if TYPE_CHECKING:
    from ..llm import LlmModel

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


def relationship_reasoning_prompt(
    relationship: RelationshipView,
    memories: list[MemoryView] | tuple[MemoryView, ...],
    user_name: str = "CurrentUser",
) -> str:
    """Build the user prompt for a relationship projection refresh."""

    return (
        f"The fixed current user is named {user_name!r}; the current user is not "
        "an endpoint Person. Rebuild the projection for the relationship between "
        "the two endpoint People in the input. Linked memories indicate relevance, "
        "not relationship evidence; summarize only interactions between the endpoints. "
        "Use only the supplied context; do not transfer standalone traits, preferences, "
        "or actions into a relationship claim.\n\nTarget relationship:\n"
        + _json(relationship)
        + "\n\nRelevant memories:\n"
        + _json(list(memories))
    )


class RelationshipReasoner:
    """Refresh one stale Relationship projection per attempt.

    Mirrors `PersonReasoner`: an in-flight watermark conflict simply causes
    the next `attempt` to recompute from the latest snapshot.
    """

    name = "relationship"

    def __init__(self, store: WorldStore, model: LlmModel) -> None:
        self.store = store
        self.model = model

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
        result = await self.model.reason_relationship(
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


__all__ = [
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "RelationshipReasoner",
    "relationship_reasoning_prompt",
]
