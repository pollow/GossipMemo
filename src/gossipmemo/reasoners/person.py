"""Person profile-card reasoner."""

from __future__ import annotations

from ..llm import LlmModel
from ..queue import ReasonerCallQueue
from ..store import WorldStore


class PersonReasoner:
    """Refresh one stale Person card per attempt.

    If Extract updates the same Person while an LLM call is in flight, the
    optimistic watermark check in `apply_person_reasoning` fails and the next
    `attempt` recomputes from the latest snapshot without taking a lock.
    """

    name = "person"

    def __init__(self, store: WorldStore, model: LlmModel, queue: ReasonerCallQueue) -> None:
        self.store = store
        self.model = model
        self.queue = queue

    async def attempt(self, space_id: str) -> bool:
        people, _, _ = self.store.stale_entities()
        targets = [person_id for sid, person_id in people if sid == space_id]
        if not targets:
            return False
        person_id = targets[0]
        more_targets = len(targets) > 1
        context = self.store.person_context(space_id, person_id)
        if not context:
            return more_targets
        person, memories, watermark = context
        if not person.stale:
            return more_targets
        inferred, hypotheses = self.store.owner_review_context(space_id, "person", person_id)
        result = await self.queue.submit(
            "reason-person", self.model.reason_person, person, memories, inferred, hypotheses
        )
        applied = self.store.apply_person_reasoning(
            space_id,
            person_id,
            watermark,
            result,
            {memory.id for memory in inferred},
            {hypothesis.id for hypothesis in hypotheses},
        )
        return more_targets or not applied


__all__ = ["PersonReasoner"]
