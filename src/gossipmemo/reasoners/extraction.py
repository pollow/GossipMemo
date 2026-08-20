"""Extraction reasoner and its prompt.

Batch *creation* (bucketing raw messages, the 30-minute partial-flush timer,
the ingest-time trigger) stays in `world.py`; that is scheduling, the
driver's job. This reasoner only consumes pending batches: each `attempt`
runs the single oldest pending batch for a space through to completion or
failure and reports whether another batch remains.

Batches drain sequentially per space, which matches the intended
"extraction drains before induction" priority (`TIER_FRESHNESS`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient, embed_query_vector
from ..models import ExtractionResult, MemoryView, ModelMessage
from ..priority import TIER_FRESHNESS, current_call_label, current_call_tier, llm_call_tier
from ..prompts import PromptLibrary, schema_instruction
from ..prompts.render import _json, fill
from ..store import PendingExtraction, WorldStore
from ..transport import ChatMessage, LlmTransport, structured
from .base import AttemptLoop, Reasoner
from .settings import ReasoningSettings

logger = logging.getLogger(__name__)

# A batch that keeps failing is skipped once it hits this many attempts, so a
# permanently-broken batch cannot spin the drain loop forever alongside other
# permanently-broken batches (each failure re-sorts the others ahead of it in
# `pending_extractions`, which alone does not bound the loop).
MAX_EXTRACTION_ATTEMPTS = 5


def _is_exhausted(pending: PendingExtraction) -> bool:
    """Has this batch spent its attempts on failures we actually saw?

    `mark_extraction_attempt` bumps the count *before* the model call, so a
    process killed mid-call (SIGKILL past Docker's stop grace period)
    leaves the count raised with the batch still 'pending' -- it neither
    succeeded nor failed, and nobody recorded why. Counting those would let
    repeated restarts retire a perfectly healthy batch after five kills.
    Only a batch that reached 'failed', with a recorded error, is allowed
    to run out of attempts.
    """
    return (
        pending.state == "failed" and pending.attempts >= MAX_EXTRACTION_ATTEMPTS
    )


def extraction_prompt(
    messages: list[ModelMessage],
    context: list[ModelMessage] | tuple[ModelMessage, ...] = (),
    known_people: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    user_name: str = "CurrentUser",
    comparison_memories: list[MemoryView] | tuple[MemoryView, ...] = (),
    *, prompts: PromptLibrary,
) -> str:
    """Build the user prompt for :class:`~models.ExtractionResult`.

    The instruction paragraphs come from the library; which messages count as
    evidence, and how each section is rendered, stays here -- that split is
    what keeps an override from reshaping the request.
    """

    evidence_messages = [message for message in messages if message.author == "user"]
    assistant_messages = [
        message for message in messages if message.author == "assistant"
    ]
    return (
        prompts.extraction_retention_rule
        + "\n"
        + fill(prompts.extraction_user_evidence_rule, quoted_user_name=repr(user_name))
        + "\n"
        + fill(prompts.extraction_person_identity_rule, user_name=user_name)
        + " "
        + prompts.extraction_time_bound_rule
        + "\n\nRecent context (context only):\n"
        + _json(list(context))
        + "\n\nKnown people (identity hints only):\n"
        + _json(list(known_people))
        + "\n"
        + prompts.extraction_known_people_rule
        + "\n"
        + "\n\nCurrent batch evidence "
        "(user-authored; the only messages allowed to produce memories):\n"
        + _json(evidence_messages)
        + "\n\nCurrent batch context (assistant-authored; context only):\n"
        + _json(assistant_messages)
        + "\n\nComparison memories (deduplication/update reference only; never new evidence):\n"
        + _json(list(comparison_memories))
        + "\n"
        + prompts.extraction_comparison_rule
    )


async def _extract(
    transport: LlmTransport,
    settings: ReasoningSettings,
    messages: Sequence[ModelMessage],
    context: Sequence[ModelMessage] = (),
    known_people: Sequence[dict[str, Any]] = (),
    comparison_memories: Sequence[MemoryView] = (),
) -> ExtractionResult:
    request = transport.prepare(
        [
            ChatMessage(
                role="system",
                content=settings.prompts.extraction_system + "\n\n"
                + schema_instruction(ExtractionResult),
            ),
            ChatMessage(
                role="user",
                content=extraction_prompt(
                    list(messages), list(context), list(known_people),
                    settings.user_name, list(comparison_memories),
                    prompts=settings.prompts,
                ),
            ),
        ],
        structured=True,
    )
    _, result = await structured(
        transport, request.messages, ExtractionResult,
        tier=current_call_tier(), label=current_call_label(),
    )
    return result


class _ExtractionReasoner(AttemptLoop):
    """Run one pending extraction batch per attempt.

    Unlike the descriptor-shaped reasoners, a failed model call or apply
    must not propagate: it is recorded on the batch (`fail_extraction`) and
    logged, matching the pre-existing behavior of `SocialMemoryWorld._extract`.
    """

    name = "extraction"

    def __init__(
        self, store: WorldStore, model: LlmTransport, settings: ReasoningSettings,
        embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
        embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store
        self.transport = model
        self.settings = settings
        # A getter rather than the client itself: this reasoner is built once
        # in `SocialMemoryWorld.__init__`, before `start()` resolves the
        # embedding client asynchronously -- see `world._embedding_client`.
        self._embedding_client_getter = embedding_client_getter or (lambda: None)
        self._embedding_query_timeout_seconds = embedding_query_timeout_seconds

    async def _comparison_query_vectors(
        self, texts: Sequence[str],
    ) -> dict[str, list[float]] | None:
        """Embed each distinct user-authored text with the "same fact?"
        instruction, for `load_extraction_comparisons`'s hybrid recall.

        Keyed by text (not position) -- see that method's docstring for why.
        `None` (not an empty dict) when embedding is unavailable at all, so
        the store takes its exact pre-existing FTS-only path rather than a
        hybrid path with zero vectors.
        """
        client = self._embedding_client_getter()
        if client is None:
            return None
        vectors: dict[str, list[float]] = {}
        for text in dict.fromkeys(texts):  # de-duplicate, preserve order
            vector = await embed_query_vector(
                client, text,
                instruction=self.settings.prompts.embedding_extraction_comparison_instruction,
                timeout=self._embedding_query_timeout_seconds,
            )
            if vector is not None:
                vectors[text] = vector
        return vectors

    async def _attempt(self, space_id: str) -> bool:
        eligible = [
            pending.batch_id
            for pending in self.store.pending_extractions()
            if pending.space_id == space_id
            and pending.batch_id is not None
            and not _is_exhausted(pending)
        ]
        if not eligible:
            return False
        batch_id = eligible[0]
        more_batches = len(eligible) > 1
        messages = self.store.load_batch(space_id, batch_id)
        if not messages:
            return more_batches
        context = self.store.load_extraction_context(space_id, batch_id)
        known_people = self.store.load_known_people(space_id, messages + context)
        comparison_texts = [
            message.content for message in messages if message.author == "user"
        ]
        query_vectors = await self._comparison_query_vectors(comparison_texts)
        comparisons = self.store.load_extraction_comparisons(
            space_id, batch_id, query_vectors=query_vectors,
        )
        started = asyncio.get_running_loop().time()
        self.store.mark_extraction_attempt(space_id, batch_id)
        try:
            with llm_call_tier(TIER_FRESHNESS, "extract"):
                result = await _extract(
                    self.transport, self.settings, messages, context, known_people, comparisons,
                )
            self.store.apply_extraction(
                space_id, batch_id, result,
                {memory.id for memory in comparisons},
            )
            logger.info(
                "extraction_completed",
                extra={
                    "space_id": space_id,
                    "batch_id": batch_id,
                    "message_count": len(messages),
                    "duration_ms": round(
                        (asyncio.get_running_loop().time() - started) * 1000, 2
                    ),
                },
            )
        except Exception as error:
            self.store.fail_extraction(space_id, batch_id, str(error))
            logger.exception("extract failed for %s", batch_id)
            return more_batches
        return more_batches


def build_extraction_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> Reasoner:
    return _ExtractionReasoner(
        store, model, settings, embedding_client_getter, embedding_query_timeout_seconds,
    )


__all__ = ["build_extraction_reasoner", "extraction_prompt"]
