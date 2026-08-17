"""Person profile-card reasoner and its prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from ..models import HypothesisView, MemoryView, PersonProjectionResult, PersonReasoningResult, PersonView
from ..store import WorldStore
from .base import DescriptorReasoner
from .owner import owner_reasoning

if TYPE_CHECKING:
    from ..llm import LlmTransport

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


@dataclass(slots=True)
class _PersonTarget:
    """One candidate Person, plus whether other stale targets remain.

    `person` is `None` when the chosen target vanished between the stale
    scan and the context read, or is no longer stale by the time it is
    read -- either way, `call` skips the model call for this attempt.
    """

    person_id: str
    more_targets: bool
    person: PersonView | None = None
    memories: tuple[MemoryView, ...] = ()
    watermark: object = None
    inferred: tuple[MemoryView, ...] = field(default_factory=tuple)
    hypotheses: tuple[HypothesisView, ...] = field(default_factory=tuple)


async def _reason_person(
    transport: "LlmTransport", person: PersonView, memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
) -> PersonReasoningResult:
    projection, actions = await owner_reasoning(
        transport, PERSON_REASONING_SYSTEM_PROMPT, person, memories, inferred_memories,
        hypotheses, PersonProjectionResult,
    )
    return PersonReasoningResult(
        profile_card=projection.profile_card, **actions.model_dump(exclude_none=True),
    )


def build_person_reasoner(store: WorldStore, model: "LlmTransport") -> DescriptorReasoner:
    """Refresh one stale Person card per attempt.

    If Extract updates the same Person while an LLM call is in flight, the
    optimistic watermark check in `apply_person_reasoning` fails and the next
    `attempt` recomputes from the latest snapshot without taking a lock.
    """

    reason_person = partial(_reason_person, model)

    def load_context(space_id: str) -> _PersonTarget | None:
        people, _, _ = store.stale_entities()
        targets = [person_id for sid, person_id in people if sid == space_id]
        if not targets:
            return None
        person_id = targets[0]
        more_targets = len(targets) > 1
        context = store.person_context(space_id, person_id)
        if not context:
            return _PersonTarget(person_id, more_targets)
        person, memories, watermark = context
        if not person.stale:
            return _PersonTarget(person_id, more_targets)
        inferred, hypotheses = store.owner_review_context(
            space_id, "person", person_id)
        return _PersonTarget(
            person_id, more_targets, person, tuple(memories), watermark,
            tuple(inferred), tuple(hypotheses),
        )

    def call(space_id: str, context: _PersonTarget):
        if context.person is None:
            return None
        return (
            "reason-person",
            reason_person,
            (context.person, context.memories,
             context.inferred, context.hypotheses),
        )

    def apply(space_id: str, context: _PersonTarget, result) -> bool:
        return store.apply_person_reasoning(
            space_id,
            context.person_id,
            context.watermark,
            result,
            {memory.id for memory in context.inferred},
            {hypothesis.id for hypothesis in context.hypotheses},
        )

    def continue_when(context: _PersonTarget, result, applied: bool) -> bool:
        return context.more_targets or not applied

    return DescriptorReasoner("person", load_context, call, apply, continue_when)


__all__ = ["PERSON_REASONING_SYSTEM_PROMPT", "build_person_reasoner"]
