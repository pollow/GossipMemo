"""Person profile-card reasoner and its prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import MemoryView, PersonView
from ..prompts import _json
from ..store import WorldStore

if TYPE_CHECKING:
    from ..llm import LlmModel

PERSON_REASONING_SYSTEM_PROMPT = """Reason carefully about one named Person from the
supplied owner context. Linked memories indicate
relevance to the target, not that the target is the semantic subject of every
memory. Do not transfer the current user's or any co-occurring person's traits,
preferences, intentions, or actions onto the target. Recurring patterns require
multiple distinct source memories; a narrow impression from one highly diagnostic
event is allowed when calibrated to that evidence. Identify supported patterns in behavior, preferences,
communication, decision-making, sensitivities, and helpful ways to interact.
Make reasonable social inferences when supported, with uncertainty proportional
to evidence. Do not use projections, inferred memories, or hypotheses as evidence.
Distinguish current conditions from historical events. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""


def person_reasoning_prompt(
    person: PersonView,
    memories: list[MemoryView] | tuple[MemoryView, ...],
    user_name: str = "CurrentUser",
) -> str:
    """Build the user prompt for a person projection refresh."""

    return (
        f"The fixed current user is named {user_name!r}; the current user is not "
        "the target Person. Rebuild the profile card for the target Person from "
        "the active memories. Linked memories indicate relevance, not semantic "
        "subject. Attribute traits, preferences, and actions to the target only "
        "when the memory supports that attribution. Use only the supplied context; "
        "prefer concise traits, preferences, current_state, and interaction_notes "
        f"that help {user_name} socialize.\n\nTarget Person:\n"
        + _json(person)
        + "\n\nActive memories:\n"
        + _json(list(memories))
    )


class PersonReasoner:
    """Refresh one stale Person card per attempt.

    If Extract updates the same Person while an LLM call is in flight, the
    optimistic watermark check in `apply_person_reasoning` fails and the next
    `attempt` recomputes from the latest snapshot without taking a lock.
    """

    name = "person"

    def __init__(self, store: WorldStore, model: LlmModel) -> None:
        self.store = store
        self.model = model

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
        result = await self.model.reason_person(person, memories, inferred, hypotheses)
        applied = self.store.apply_person_reasoning(
            space_id,
            person_id,
            watermark,
            result,
            {memory.id for memory in inferred},
            {hypothesis.id for hypothesis in hypotheses},
        )
        return more_targets or not applied


__all__ = ["PERSON_REASONING_SYSTEM_PROMPT", "PersonReasoner", "person_reasoning_prompt"]
