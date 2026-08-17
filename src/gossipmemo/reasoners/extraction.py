"""Extraction reasoner and its prompt.

Batch *creation* (bucketing raw messages, the 30-minute partial-flush timer,
the ingest-time trigger) stays in `world.py`; that is scheduling, the
driver's job. This reasoner only consumes pending batches: each `attempt`
runs the single oldest pending batch for a space through to completion or
failure and reports whether another batch remains.

ACCEPTED INTENTIONAL CHANGE: batches used to run as one spawned task each,
concurrently. Driven as a reasoner they now drain sequentially per space.
Provider requests were already serialized by `ProviderGate`, so this is a
scheduling change, not a semantic one, and it matches the intended
"extraction drains before induction" priority (`TIER_FRESHNESS`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..models import MemoryView, ModelMessage
from ..priority import TIER_FRESHNESS, llm_call_tier
from ..prompts import _json
from ..store import PendingExtraction, WorldStore
from .base import AttemptLoop, Reasoner

if TYPE_CHECKING:
    from ..llm import LlmModel

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

EXTRACTION_SYSTEM_PROMPT = """Extract useful, provenance-aware memories from the messages.
Return only the supplied JSON schema. Keep the original meaning, speaker, and
uncertainty. Extract explicit facts, events, preferences, plans, and situations;
do not make broad personality inferences from one conversation. Leave recurring
patterns to reasoning passes. Use stated, observed, or reported basis; never use
an inferred basis for extraction. Resolve relative dates using each message's
occurred_at. A reported claim must remain attributed. The user is not a Person;
two people appearing
together do not by themselves establish a relationship. Use the language that
best matches the dominant language of current user evidence for every generated
natural-language field, including new display names; keep IDs and enum values unchanged.
"""


def extraction_prompt(
    messages: list[ModelMessage],
    policy: str = "balanced",
    context: list[ModelMessage] | tuple[ModelMessage, ...] = (),
    known_people: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    user_name: str = "CurrentUser",
    comparison_memories: list[MemoryView] | tuple[MemoryView, ...] = (),
) -> str:
    """Build the user prompt for :class:`~models.ExtractionResult`."""

    policy_text = {
        "conservative": (
            "Keep only explicit, durable facts and meaningful events; skip "
            "isolated transient details such as 'today I had coffee'."
        ),
        "balanced": (
            "Keep explicit durable information and a transient detail only when "
            "it affects an ongoing situation or helps reveal a recurring pattern."
        ),
        "comprehensive": (
            "Keep explicit information plus relevant short-lived events as dated "
            "evidence, including isolated details that may reveal a later pattern."
        ),
    }
    if policy not in policy_text:
        raise ValueError(f"unknown extraction policy: {policy}")
    evidence_messages = [message for message in messages if message.author == "user"]
    assistant_messages = [
        message for message in messages if message.author == "assistant"
    ]
    return (
        f"Use the server's {policy} extraction policy for the whole batch.\n"
        f"{policy_text[policy]}\n"
        f"The fixed current user is named {user_name!r}. Use that name when referring "
        "to the current user; the current user is not a Person. User-authored "
        "messages in the current batch are the only evidence allowed to create "
        "memories. Assistant-authored messages in the batch and recent context are "
        "context only: use them to resolve references and conversational meaning, "
        "but never save their restatements, summaries, analyses, or advice as new "
        "memories. Assistant content may supply a proposition only when a current "
        "user evidence message explicitly confirms, adopts, or corrects it.\n"
        "The current user/assistant author role is context and never a Person. List every "
        "specific, external human individual referenced by a memory in its `people` "
        "refs; express who said or did what in the memory content itself. A Person "
        "must denote one concrete human whose identity is distinguishable from "
        "other people in the evidence and could remain recognizable across "
        "conversations. It needs an evidence-supported durable identity anchor: a "
        "proper name, an explicit alias, or a role whose single holder is uniquely "
        "and temporally determined by the evidence. Do not treat a grammatical "
        "anaphor, an unbounded group or category, a non-human entity, or a merely "
        "situational description as an identity. For an unnamed but sufficiently "
        "anchored individual, choose the most stable, specific, neutral canonical "
        "label supported by the evidence, in the evidence language, and keep each "
        "observed surface wording as an alias. Do not create separate identities "
        "merely because synonymous wording was used. Never guess that differently named people are the "
        "same person. If identity is vague, preserve any otherwise useful durable "
        "memory with the original reference in its content and leave its `people` "
        "refs empty; identity uncertainty alone is not a reason to discard that "
        "memory. Preserve reported claims as "
        "`reported`, not facts. Set `about_user` for a claim or event about "
        f"{user_name}; {user_name} must never appear in "
        "`people`. "
        "Record valid_from/valid_to when the message gives a time bound.\n\n"
        "Recent context (context only):\n"
        + _json(list(context))
        + "\n\nKnown people (identity hints only):\n"
        + _json(list(known_people))
        + "\nIn natural-language memory content and generated display fields, use "
        "a known person's canonical display_name when the messages refer to them. "
        "In every ExtractedMemory.people and ExtractedRelationship.person_a_ref/"
        "person_b_ref, use the supplied stable Person `id` (never a display_name "
        "or alias). "
        "If the messages explicitly introduce a new short name, return "
        "it in that person's `aliases` field. Omit a known person unless a new "
        "memory references them or the messages explicitly add an alias. Do not "
        "echo the known-people list.\n"
        + "\n\nCurrent batch evidence (user-authored; the only messages allowed to produce memories):\n"
        + _json(evidence_messages)
        + "\n\nCurrent batch context (assistant-authored; context only):\n"
        + _json(assistant_messages)
        + "\n\nComparison memories (deduplication/update reference only; never new evidence):\n"
        + _json(list(comparison_memories))
        + "\nFor comparison memories only: omit a memory when the current user batch merely repeats it. "
        "When current user evidence explicitly corrects, updates, or refines one, emit the new memory "
        "and set `supersedes_memory_id` to that supplied comparison memory ID. Never copy details from a "
        "comparison memory unless those details also appear in current user evidence. Do not use an inferred "
        "comparison memory as evidence."
    )


class _ExtractionReasoner(AttemptLoop):
    """Run one pending extraction batch per attempt.

    Unlike the descriptor-shaped reasoners, a failed model call or apply
    must not propagate: it is recorded on the batch (`fail_extraction`) and
    logged, matching the pre-existing behavior of `SocialMemoryWorld._extract`.
    """

    name = "extraction"

    def __init__(self, store: WorldStore, model: LlmModel) -> None:
        self.store = store
        self.model = model

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
        comparisons = self.store.load_extraction_comparisons(space_id, batch_id)
        started = asyncio.get_running_loop().time()
        self.store.mark_extraction_attempt(space_id, batch_id)
        try:
            with llm_call_tier(TIER_FRESHNESS, "extract"):
                result = await self.model.extract(
                    messages, context, known_people, comparisons,
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


def build_extraction_reasoner(store: WorldStore, model: LlmModel) -> Reasoner:
    return _ExtractionReasoner(store, model)


__all__ = ["EXTRACTION_SYSTEM_PROMPT", "build_extraction_reasoner", "extraction_prompt"]
