"""The two-phase owner-reasoning fold shared by person, relationship, and
user_model.

One fold step is `(current card, one batch of memories) -> new card`. Phase
1 asks for the projection (profile card / relationship facets) alone;
phase 2 replays that exact projection as an assistant turn and asks only
for inferred-memory and hypothesis actions against it. That split lets
`structured()` validate the projection and the actions independently
against their own schemas, and lets the check below prove the *actions*
request (the larger of the two, since it embeds the first completion)
fits before any HTTP call is made.

The caller supplies a delta -- memories newer than the card's watermark,
including the ones that are no longer active -- so steady-state
maintenance is one fold step over a handful of rows. A card with no
watermark yet gets the whole history instead, folded batch by batch into
the card the previous batch produced; that iteration is why an oversized
prompt no longer has to be compressed lossily. Batches come from
`chunking.greedy_chunks`, so a batch is as large as the budget allows and
a single oversized memory is split by content rather than summarized
away.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from ..chunking import greedy_chunks
from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient
from ..models import (
    HypothesisView,
    MemoryView,
)
from ..priority import current_call_label, current_call_tier
from ..prompts import (
    actions_stage_prompt,
    owner_reasoning_prefix,
    projection_stage_prompt,
    schema_instruction,
)
from ..store import WorldStore
from ..transport import (
    ChatMessage,
    LlmTransport,
    structured,
)
from .dedup_priority import similarity_priority_order
from .settings import ReasoningSettings

logger = logging.getLogger(__name__)

ProjectionT = TypeVar("ProjectionT", bound=BaseModel)
ActionsT = TypeVar("ActionsT", bound=BaseModel)


async def owner_reasoning(
    transport: LlmTransport,
    settings: ReasoningSettings,
    system_prompt: str,
    target: BaseModel,
    memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView],
    hypotheses: Sequence[HypothesisView],
    projection_type: type[ProjectionT],
    actions_type: type[ActionsT],
    *,
    store: WorldStore | None = None,
    space_id: str | None = None,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> tuple[ProjectionT, ActionsT]:
    """Fold `memories` into `target` and return the final card plus actions.

    Comparison-only inferred memories and hypotheses are bounded first
    (they may never grow to consume the evidence budget); what is left of
    the budget decides how many memories one batch carries. Each batch
    runs the full two-call pair against the card the previous batch
    produced, and the actions of every batch are concatenated: omission
    stays a no-op, so a later batch never withdraws an earlier one's
    action.

    Before bounding, `hypotheses` is stable-resorted by similarity to
    `memories` (the evidence this call is actually reasoning about): a
    priority signal only (see `dedup_priority`), so that the hypotheses
    most likely to already express the same claim as what the evidence
    suggests are the ones `_bounded_comparisons` keeps in full text rather
    than downgrading to an ID-only skeleton. `store`/`space_id` are
    optional -- when either is missing, or embedding is unavailable, the
    order is exactly what it was before this call.
    """

    if store is not None and space_id is not None and hypotheses:
        hypotheses = await similarity_priority_order(
            store, space_id, "hypothesis", hypotheses, lambda item: item.id,
            "\n".join(memory.content for memory in memories),
            embedding_client_getter=embedding_client_getter or (lambda: None),
            instruction=settings.prompts.embedding_hypothesis_dedup_instruction,
            timeout=embedding_query_timeout_seconds,
        )

    bounded_inferred, bounded_hypotheses = _bounded_comparisons(
        transport, settings, system_prompt, target, inferred_memories, hypotheses,
        projection_type, actions_type,
    )

    def fits(batch: Sequence[MemoryView], card: BaseModel) -> bool:
        return _stage2_fits(
            transport, settings,
            _first_messages(
                transport, settings, system_prompt, card, batch, bounded_inferred,
                bounded_hypotheses, projection_type,
            ),
            actions_type,
        )

    def split(batch: Sequence[MemoryView], card: BaseModel) -> list[list[MemoryView]]:
        def check(candidate: Sequence[MemoryView]) -> None:
            if not fits(candidate, card):
                raise ValueError("owner evidence batch exceeds context budget")

        return greedy_chunks(list(batch), lambda candidate: fits(candidate, card), check)

    # An owner with no evidence at all still gets one pass: the card is
    # rewritten from the comparison-only context it does have.
    pending = deque(split(memories, target) or [[]])
    card: BaseModel = target
    projection: ProjectionT | None = None
    collected: list[ActionsT] = []
    while pending:
        batch = pending.popleft()
        if not fits(batch, card):
            # Folding earlier batches grew the card, so this batch no
            # longer fits beside it. Re-pack it against the current card.
            pieces = split(batch, card)
            if len(pieces) < 2:
                raise ValueError("owner evidence batch exceeds context budget")
            pending.extendleft(reversed(pieces))
            continue
        first_messages = _first_messages(
            transport, settings, system_prompt, card, batch, bounded_inferred,
            bounded_hypotheses, projection_type,
        )
        first, projection = await structured(
            transport, first_messages, projection_type,
            tier=current_call_tier(), label=current_call_label(),
        )
        _, actions = await structured(
            transport,
            first_messages + [
                ChatMessage(role="assistant", content=first),
                ChatMessage(
                    role="user",
                    content=actions_stage_prompt(settings.prompts) + "\n"
                    + schema_instruction(actions_type),
                ),
            ],
            actions_type,
            tier=current_call_tier(), label=current_call_label(),
        )
        collected.append(actions)
        card = card.model_copy(update=projection.model_dump())
    if projection is None:
        raise ValueError("owner reasoning ran no fold step")
    return projection, _merged_actions(collected, actions_type)


def _merged_actions(results: Sequence[ActionsT], actions_type: type[ActionsT]) -> ActionsT:
    """Concatenate the per-batch action results into one.

    Every field of an actions result is an optional sub-model whose own
    fields are lists of independent, explicitly scoped items, so a merge
    is a concatenation: order is preserved, and storage applies the same
    per-item validation it would have applied to a single batch's result.
    """

    if len(results) == 1:
        return results[0]
    merged: dict[str, Any] = {}
    for name in actions_type.model_fields:
        parts = [part for part in (getattr(item, name) for item in results) if part is not None]
        if not parts:
            continue
        combined: dict[str, Any] = {}
        for part in parts:
            for field, value in part:
                if isinstance(value, list):
                    combined.setdefault(field, []).extend(value)
                else:
                    combined[field] = value
        merged[name] = type(parts[0])(**combined)
    return actions_type(**merged)


def _first_messages(
    transport: LlmTransport, settings: ReasoningSettings, system_prompt: str,
    target: BaseModel, evidence: Sequence[MemoryView], inferred: Sequence[MemoryView],
    hypotheses: Sequence[HypothesisView], projection_type: type[BaseModel],
) -> list[ChatMessage]:
    prefix = owner_reasoning_prefix(
        target,
        [memory for memory in evidence if memory.status == "active"],
        [memory for memory in evidence if memory.status != "active"],
        list(inferred), list(hypotheses),
        prompts=settings.prompts, user_name=settings.user_name,
    )
    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=prefix),
        ChatMessage(
            role="user",
            content=projection_stage_prompt(settings.prompts) + "\n"
            + schema_instruction(projection_type),
        ),
    ]


def _stage2_estimate(
    transport: LlmTransport, settings: ReasoningSettings, first_messages: list[ChatMessage],
    actions_type: type[BaseModel],
) -> int:
    request = transport.prepare(
        first_messages + [
            ChatMessage(role="assistant", content=""),
            ChatMessage(
                role="user",
                content=actions_stage_prompt(settings.prompts) + "\n"
                + schema_instruction(actions_type),
            ),
        ],
        structured=True,
    )
    # The second request contains the first completion as an assistant
    # message. Reserve its configured maximum in addition to the normal
    # output reserve already excluded by ContextBudget.
    return (
        transport.context_budget.estimate_request(request)
        + transport.context_budget.output_reserve_tokens
    )


def _stage2_fits(
    transport: LlmTransport, settings: ReasoningSettings, first_messages: list[ChatMessage],
    actions_type: type[BaseModel],
) -> bool:
    return transport.context_budget.report(
        _stage2_estimate(transport, settings, first_messages, actions_type)
    ).fits


def _bounded_comparisons(
    transport: LlmTransport, settings: ReasoningSettings, system_prompt: str, target: BaseModel,
    inferred: Sequence[MemoryView], hypotheses: Sequence[HypothesisView],
    projection_type: type[BaseModel], actions_type: type[BaseModel],
) -> tuple[list[MemoryView], list[HypothesisView]]:
    """Bound comparison-only state without turning it into evidence.

    IDs are retained before prose. If even every ID skeleton cannot fit,
    the oldest tail is omitted; omission remains a no-op by contract.
    Comparison state gets at most one third of the usable input budget so
    every fold batch still has room for real evidence.
    """
    context_budget = transport.context_budget
    empty_first = _first_messages(
        transport, settings, system_prompt, target, [], [], [], projection_type)
    base = _stage2_estimate(transport, settings, empty_first, actions_type)
    ceiling = min(
        context_budget.usable_input_tokens,
        base + context_budget.usable_input_tokens // 3,
    )
    if base > ceiling:
        raise ValueError("owner prompt scaffolding exceeds context budget")

    chosen_inferred: list[MemoryView] = []
    chosen_hypotheses: list[HypothesisView] = []

    def fits(
        candidate_inferred: Sequence[MemoryView],
        candidate_hypotheses: Sequence[HypothesisView],
    ) -> bool:
        first = _first_messages(
            transport, settings, system_prompt, target, [], candidate_inferred,
            candidate_hypotheses, projection_type,
        )
        return _stage2_estimate(transport, settings, first, actions_type) <= ceiling

    # Preserve actionable IDs first with empty prose, in the stable store
    # order (newest first). Then spend remaining budget on bounded prose.
    for item in inferred:
        skeleton = item.model_copy(update={"content": ""})
        if fits([*chosen_inferred, skeleton], chosen_hypotheses):
            chosen_inferred.append(skeleton)
    for hypothesis in hypotheses:
        hypothesis_skeleton = hypothesis.model_copy(update={"content": ""})
        if fits(chosen_inferred, [*chosen_hypotheses, hypothesis_skeleton]):
            chosen_hypotheses.append(hypothesis_skeleton)

    inferred_by_id = {item.id: item for item in inferred}
    for index, skeleton in enumerate(tuple(chosen_inferred)):
        original = inferred_by_id[skeleton.id]
        expanded = original.model_copy(update={"content": original.content[:1200]})
        candidate = [*chosen_inferred]
        candidate[index] = expanded
        if fits(candidate, chosen_hypotheses):
            chosen_inferred = candidate

    hypotheses_by_id = {item.id: item for item in hypotheses}
    for index, hypothesis_skeleton in enumerate(tuple(chosen_hypotheses)):
        hypothesis_original = hypotheses_by_id[hypothesis_skeleton.id]
        hypothesis_expanded = hypothesis_original.model_copy(
            update={"content": hypothesis_original.content[:1200]})
        hypothesis_candidate = [*chosen_hypotheses]
        hypothesis_candidate[index] = hypothesis_expanded
        if fits(chosen_inferred, hypothesis_candidate):
            chosen_hypotheses = hypothesis_candidate
    return chosen_inferred, chosen_hypotheses


__all__ = ["owner_reasoning"]
