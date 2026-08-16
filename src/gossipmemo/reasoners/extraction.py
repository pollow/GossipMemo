"""Extraction prompts.

Extraction itself stays outside the `Reasoner` abstraction (see
`SocialMemoryWorld._extract`); this module only co-locates its prompt text
with the rest of the extraction-specific handling.
"""

from __future__ import annotations

from typing import Any

from ..models import MemoryView, ModelMessage
from ..prompts import _json

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


__all__ = ["EXTRACTION_SYSTEM_PROMPT", "extraction_prompt"]
