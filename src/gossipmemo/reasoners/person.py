"""Person profile-card reasoner and its prompts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient
from ..models import (
    HypothesisView,
    MemoryView,
    PersonProjectionResult,
    PersonReasoningResult,
    PersonView,
    ReasoningActionsResult,
)
from ..store import WorldStore
from .base import DescriptorReasoner
from .owner import owner_reasoning
from .settings import ReasoningSettings

if TYPE_CHECKING:
    from ..llm import LlmTransport


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
    watermark: str | None = None
    inferred: tuple[MemoryView, ...] = field(default_factory=tuple)
    hypotheses: tuple[HypothesisView, ...] = field(default_factory=tuple)


async def _reason_person(
    transport: LlmTransport, settings: ReasoningSettings, person: PersonView,
    memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    *,
    store: WorldStore | None = None,
    space_id: str | None = None,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> PersonReasoningResult:
    projection, actions = await owner_reasoning(
        transport, settings, settings.prompts.person_reasoning_system, person, memories,
        inferred_memories, hypotheses, PersonProjectionResult, ReasoningActionsResult,
        store=store, space_id=space_id, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )
    return PersonReasoningResult(
        profile_card=projection.profile_card, **actions.model_dump(exclude_none=True),
    )


def build_person_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> DescriptorReasoner:
    """Refresh one stale Person card per attempt.

    If Extract updates the same Person while an LLM call is in flight, the
    optimistic watermark check in `apply_person_reasoning` fails and the next
    `attempt` recomputes from the latest snapshot without taking a lock.
    """

    reason_person = partial(
        _reason_person, model, settings,
        store=store, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )

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
        inferred, hypotheses = store.owner_review_context(space_id, "person", person_id)
        return _PersonTarget(
            person_id, more_targets, person, tuple(memories), watermark,
            tuple(inferred), tuple(hypotheses),
        )

    def call(space_id: str, context: _PersonTarget):
        if context.person is None:
            return None
        return (
            "reason-person",
            partial(reason_person, space_id=space_id),
            (context.person, context.memories, context.inferred, context.hypotheses),
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


__all__ = ["build_person_reasoner"]
