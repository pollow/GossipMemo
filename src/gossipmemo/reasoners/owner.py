"""Two-phase owner-reasoning skeleton shared by person, relationship, and
user_model.

Phase 1 asks for the projection (profile card / relationship facets)
alone; phase 2 replays that exact projection as an assistant turn and asks
only for inferred-memory and hypothesis actions against it. That split
lets `structured()` validate the projection and the actions independently
against their own schemas, and lets the digest fallback below prove the
*actions* request (the larger of the two, since it embeds the first
completion) fits before any HTTP call is made.

Owner cards are full snapshots, not paginated evidence, so an oversized
prompt can only be shrunk lossily: `_digest_evidence` compresses raw
Memories into `OwnerEvidenceDigestView`s via `chunking.reduce_until_fits`,
each digest call itself checked and, if needed, further chunked before
being sent.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from ..chunking import reduce_until_fits
from ..models import (
    ExtractedOwnerEvidenceDigest,
    HypothesisView,
    MemoryView,
    OwnerEvidenceDigestView,
)
from ..priority import current_call_label, current_call_tier
from ..prompts import (
    actions_stage_prompt,
    owner_evidence_digest_prompt,
    owner_reasoning_prefix,
    projection_stage_prompt,
    schema_instruction,
)
from ..transport import (
    ChatCompletionRequest,
    ChatMessage,
    LLMOutputError,
    LlmTransport,
    structured,
)
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
) -> tuple[ProjectionT, ActionsT]:
    """Run the shared two-call owner-reasoning shape over `transport`.

    Comparison-only inferred memories and hypotheses are bounded first
    (they may never grow to consume the evidence budget); if the actions
    request still would not fit with full evidence, evidence is digested
    and the whole prefix rebuilt before either call goes out.
    """

    bounded_inferred, bounded_hypotheses = _bounded_comparisons(
        transport, settings, system_prompt, target, inferred_memories, hypotheses,
        projection_type, actions_type,
    )
    first_messages = _first_messages(
        transport, settings, system_prompt, target, memories, bounded_inferred,
        bounded_hypotheses, projection_type,
    )
    if not _stage2_fits(transport, settings, first_messages, actions_type):
        digest = await _digest_evidence(
            transport, settings, target, memories, bounded_inferred, bounded_hypotheses,
            system_prompt, projection_type, actions_type,
        )
        first_messages = _first_messages(
            transport, settings, system_prompt, target, digest, bounded_inferred,
            bounded_hypotheses, projection_type,
        )
        if not _stage2_fits(transport, settings, first_messages, actions_type):
            raise ValueError("owner evidence digest did not fit context budget")

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
    return projection, actions


def _first_messages(
    transport: LlmTransport, settings: ReasoningSettings, system_prompt: str,
    target: BaseModel, evidence: Sequence[Any], inferred: Sequence[MemoryView],
    hypotheses: Sequence[HypothesisView], projection_type: type[BaseModel],
) -> list[ChatMessage]:
    prefix = owner_reasoning_prefix(
        target, list(evidence), list(inferred), list(hypotheses),
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
    evidence always has room to be represented or digested.
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


def _evidence_source_ids(item: MemoryView | OwnerEvidenceDigestView) -> set[str]:
    if isinstance(item, MemoryView):
        return {item.id}
    return set(item.source_memory_ids)


def _digest_request(
    transport: LlmTransport, settings: ReasoningSettings, chunk: list[Any]
) -> ChatCompletionRequest:
    return transport.prepare(
        [
            ChatMessage(
                role="system",
                content="Compress evidence only; do not infer people or actions.\n\n"
                + schema_instruction(ExtractedOwnerEvidenceDigest),
            ),
            ChatMessage(role="user", content=owner_evidence_digest_prompt(
                chunk, settings.user_name, prompts=settings.prompts)),
        ],
        structured=True,
    )


async def _digest_evidence(
    transport: LlmTransport, settings: ReasoningSettings, target: BaseModel,
    memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView], hypotheses: Sequence[HypothesisView],
    system_prompt: str, projection_type: type[BaseModel], actions_type: type[BaseModel],
) -> list[OwnerEvidenceDigestView]:
    context_budget = transport.context_budget

    def chunks_for(
        source: Sequence[MemoryView | OwnerEvidenceDigestView],
    ) -> list[list[MemoryView | OwnerEvidenceDigestView]]:
        chunks: list[list[MemoryView | OwnerEvidenceDigestView]] = []
        current: list[MemoryView | OwnerEvidenceDigestView] = []
        for memory in source:
            # Segment oversized content so every digest request is valid.
            pieces: list[Any] = [memory]
            if isinstance(memory, MemoryView) and not context_budget.report(
                context_budget.estimate_request(_digest_request(transport, settings, [memory]))
            ).fits:
                lo, hi = 1, len(memory.content)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    candidate_piece = memory.model_copy(update={"content": memory.content[:mid]})
                    if context_budget.report(
                        context_budget.estimate_request(
                            _digest_request(transport, settings, [candidate_piece]))
                    ).fits:
                        lo = mid
                    else:
                        hi = mid - 1
                width = lo
                if width < 1:
                    raise ValueError(
                        f"single owner evidence segment exceeds context budget: {memory.id}"
                    )
                pieces = [
                    memory.model_copy(update={"content": memory.content[i:i + width]})
                    for i in range(0, len(memory.content), width)
                ]
            for piece in pieces:
                candidate = current + [piece]
                candidate_ids = set().union(*(_evidence_source_ids(item) for item in candidate))
                source_id_limit = (
                    32 if any(isinstance(item, MemoryView) for item in candidate) else 512
                )
                candidate_fits = context_budget.report(
                    context_budget.estimate_request(_digest_request(transport, settings, candidate))
                ).fits and len(candidate_ids) <= source_id_limit
                if current and not candidate_fits:
                    chunks.append(current)
                    current = []
                current.append(piece)
                if not context_budget.report(
                    context_budget.estimate_request(_digest_request(transport, settings, current))
                ).fits:
                    identifier = getattr(memory, "id", "digest")
                    raise ValueError(
                        f"single owner evidence segment exceeds context budget: {identifier}"
                    )
        if current:
            chunks.append(current)
        return chunks

    async def digest_chunk(
        chunk: list[MemoryView | OwnerEvidenceDigestView],
    ) -> list[OwnerEvidenceDigestView]:
        request = _digest_request(transport, settings, chunk)
        context_budget.check(context_budget.estimate_request(request))
        allowed = set().union(*(_evidence_source_ids(item) for item in chunk))
        recursive = all(isinstance(item, OwnerEvidenceDigestView) for item in chunk)
        policy = transport.retry_policy
        semantic_attempt = 0
        while True:
            _, result = await structured(
                transport, request.messages, ExtractedOwnerEvidenceDigest,
                tier=current_call_tier(), label=current_call_label(),
            )
            if recursive and result.items:
                # Reduce inputs already passed strict raw-ID validation.
                # Ignore any model-copied IDs here and deterministically
                # inherit the trusted server-side union.
                return [OwnerEvidenceDigestView(
                    summary="\n".join(item.summary for item in result.items)[:600],
                    source_memory_ids=sorted(allowed),
                    basis="compressed", uncertainty="", semantic_subject="",
                )]
            accepted = [
                OwnerEvidenceDigestView.model_validate(item.model_dump(mode="json"))
                for item in result.items
                if item.source_memory_ids and set(item.source_memory_ids) <= allowed
            ]
            covered = set().union(*(set(item.source_memory_ids) for item in accepted), set())
            if covered == allowed:
                return accepted
            if semantic_attempt >= policy.attempts:
                raise LLMOutputError(
                    "owner evidence digest omitted or invented source Memory IDs"
                )
            delay = policy.delay(semantic_attempt)
            logger.warning(
                "llm_output_retry_scheduled",
                extra={
                    "attempt": semantic_attempt + 1,
                    "result_type": "ExtractedOwnerEvidenceDigest",
                    "reason": "source_memory_ids_mismatch",
                    "delay_seconds": round(delay, 3),
                },
            )
            await asyncio.sleep(delay)
            semantic_attempt += 1

    async def reduce_round(
        source: Sequence[MemoryView | OwnerEvidenceDigestView],
    ) -> list[MemoryView | OwnerEvidenceDigestView]:
        output: list[MemoryView | OwnerEvidenceDigestView] = []
        for chunk in chunks_for(source):
            output.extend(await digest_chunk(chunk))
        if not output:
            raise ValueError("owner evidence digest made no progress")
        return output

    def final_first_messages(
        output: list[MemoryView | OwnerEvidenceDigestView]
    ) -> list[ChatMessage]:
        return _first_messages(
            transport, settings, system_prompt, target, output, list(inferred_memories),
            list(hypotheses), projection_type,
        )

    def target_fits(output: list[MemoryView | OwnerEvidenceDigestView]) -> bool:
        return _stage2_fits(transport, settings, final_first_messages(output), actions_type)

    def progress_size(output: list[MemoryView | OwnerEvidenceDigestView]) -> int:
        return context_budget.estimate_text(
            owner_reasoning_prefix(
                target, output, list(inferred_memories), list(hypotheses),
                prompts=settings.prompts, user_name=settings.user_name,
            )
        )

    source: list[MemoryView | OwnerEvidenceDigestView] = list(memories)
    reduced = await reduce_until_fits(
        reduce_round, target_fits, progress_size, source,
        max_rounds=3, no_progress_message="owner evidence digest reduction made no progress",
    )
    # Every round replaces its input with digests, so the survivors are digests.
    return cast(list[OwnerEvidenceDigestView], reduced)


__all__ = ["owner_reasoning"]
