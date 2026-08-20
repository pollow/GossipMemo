"""Rolling continuity reasoner.

Deliberately not part of `DEFAULT_REASONING_PIPELINE`: continuity keeps its
own message-count trigger (`SocialMemoryWorld._schedule_continuity_reason`)
instead of running on the daily/startup induction sweep.

Continuity is an accumulator, not a snapshot (unlike the owner family in
`reasoners/owner.py`): an oversized backlog is paginated across several
calls via `chunking.greedy_chunks` rather than lossily digested. Pagination
sizes each chunk against a placeholder prior (`chunk_prior`) rather than the
real, evolving one, so chunk boundaries do not shift as the streamed prior
grows; `_fit_continuity_prior` then shrinks the real prior, if needed, to
fit alongside each chunk before that request goes out. Only the last
chunk's completion is kept, and its `through_message_id` is always
overwritten with the last message actually covered.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

from ..chunking import greedy_chunks
from ..models import ContinuityReasoningResult, ContinuityView, ModelMessage
from ..priority import TIER_FRESHNESS, current_call_label, current_call_tier
from ..prompts import _json, schema_instruction
from ..store import WorldStore
from ..transport import ChatCompletionRequest, ChatMessage, LlmTransport, structured
from .base import DescriptorReasoner
from .settings import ReasoningSettings

logger = logging.getLogger(__name__)

CONTINUITY_SYSTEM_PROMPT = """Rebuild compact cross-session continuity.
Return only the supplied JSON schema. Keep ongoing threads, recent decisions,
pending actions, and context useful for the next conversation. Do not make
long-term personality inferences or copy person/user profiles; the current user
is not a Person. Use the language that best matches supplied messages and prior
continuity; keep IDs and enum values unchanged.
"""


def continuity_prompt(
    continuity: ContinuityView | None, messages: list[ModelMessage]
) -> str:
    return (
        "Rebuild continuity from the prior summary and newer raw messages. "
        "Choose the last supplied message as through_message_id.\n\nPrior continuity:\n"
        + _json(continuity)
        + "\n\nNew messages:\n"
        + _json(messages)
    )


def _continuity_request(
    transport: LlmTransport, prior: ContinuityView | None, chunk: list[ModelMessage],
) -> ChatCompletionRequest:
    return transport.prepare(
        [
            ChatMessage(
                role="system",
                content=CONTINUITY_SYSTEM_PROMPT + "\n\n" +
                schema_instruction(ContinuityReasoningResult),
            ),
            ChatMessage(role="user", content=continuity_prompt(prior, chunk)),
        ],
        structured=True,
    )


def _fit_continuity_prior(
    transport: LlmTransport, prior: ContinuityView | None, chunk: list[ModelMessage],
    request_for: Any,
) -> ContinuityView | None:
    context_budget = transport.context_budget
    if prior is None or context_budget.report(
        context_budget.estimate_request(request_for(prior, chunk))
    ).fits:
        return prior
    text = prior.text
    lo, hi = 0, len(text)
    while lo < hi:
        middle = (lo + hi + 1) // 2
        candidate = prior.model_copy(update={"text": text[:middle]})
        if context_budget.report(
            context_budget.estimate_request(request_for(candidate, chunk))
        ).fits:
            lo = middle
        else:
            hi = middle - 1
    candidate = prior.model_copy(update={"text": text[:lo]})
    context_budget.check(context_budget.estimate_request(request_for(candidate, chunk)))
    return candidate


async def _reason_continuity(
    transport: LlmTransport, continuity: ContinuityView | None, messages: Sequence[ModelMessage],
) -> ContinuityReasoningResult:
    source = list(messages)
    if not source:
        raise ValueError("continuity requires at least one message")
    context_budget = transport.context_budget

    def request_for(
        prior: ContinuityView | None, chunk: list[ModelMessage]
    ) -> ChatCompletionRequest:
        return _continuity_request(transport, prior, chunk)

    result: ContinuityReasoningResult | None = None
    normal = request_for(continuity, source)
    if context_budget.report(context_budget.estimate_request(normal)).fits:
        _, result = await structured(
            transport, normal.messages, ContinuityReasoningResult,
            tier=current_call_tier(), label=current_call_label(),
        )
        return result

    # Reserve room for the streamed typed prior, not merely the initial
    # empty/null continuity object.
    chunk_prior = ContinuityView(
        text=(continuity.text if continuity and continuity.text else "summary"),
        related_person_ids=(
            continuity.related_person_ids if continuity and continuity.related_person_ids else [
                "person"]
        ),
        through_message_id=(
            continuity.through_message_id
            if continuity and continuity.through_message_id
            else "continuity"
        ),
    )

    def fits(chunk: Sequence[ModelMessage]) -> bool:
        return context_budget.report(
            context_budget.estimate_request(request_for(chunk_prior, list(chunk)))
        ).fits

    def check(chunk: Sequence[ModelMessage]) -> None:
        context_budget.check(context_budget.estimate_request(request_for(chunk_prior, list(chunk))))

    chunks = greedy_chunks(source, fits, check)
    # A small normal update deliberately remains one call. Large updates
    # stream typed summaries forward; only the caller persists the final one.
    prior = continuity
    for chunk in chunks:
        prior = _fit_continuity_prior(transport, prior, chunk, request_for)
        request = request_for(prior, chunk)
        _, result = await structured(
            transport, request.messages, ContinuityReasoningResult,
            tier=current_call_tier(), label=current_call_label(),
        )
        prior = ContinuityView(**result.model_dump())
    assert result is not None
    return result.model_copy(update={"through_message_id": source[-1].id})


def build_continuity_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings
) -> DescriptorReasoner:
    reason_continuity = partial(_reason_continuity, model)

    def load_context(space_id: str):
        context = store.continuity_context(space_id)
        if not context:
            return None
        _, messages = context
        if not messages:
            return None
        return context

    def call(space_id: str, context):
        continuity, messages = context
        return "reason-continuity", reason_continuity, (continuity, messages)

    def apply(space_id: str, context, result) -> bool:
        continuity, messages = context
        logger.info("continuity_extracted", extra={
                    "space_id": space_id, "message_count": len(messages)})
        expected = continuity.through_message_id if continuity else None
        return store.apply_continuity_reasoning(space_id, expected, result)

    def continue_when(context, result, applied: bool) -> bool:
        # Success means more newer messages may remain uncovered; a conflict
        # (someone else updated continuity concurrently) means give up.
        return applied

    return DescriptorReasoner(
        "continuity", load_context, call, apply, continue_when, tier=TIER_FRESHNESS,
    )


__all__ = ["CONTINUITY_SYSTEM_PROMPT", "build_continuity_reasoner", "continuity_prompt"]
