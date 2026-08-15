"""Prompt builders used by the structured LLM adapter.

Keeping prompts in a separate module gives the application one place to
version and test the model contract.  The builders return complete user
messages; the adapter supplies the corresponding system instruction and
serializes the result schema into it.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .models import (
    ExtractionResult,
    MemoryView,
    ModelMessage,
    PersonReasoningResult,
    PersonView,
    UserModelView,
    QueryContext,
    RelationshipReasoningResult,
    RelationshipView,
    ContinuityReasoningResult,
    ContinuityView,
    HypothesisView,
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


RELATIONSHIP_REASONING_SYSTEM_PROMPT = """Reason carefully about the relationship between
the two endpoint People in the supplied owner context. Linked memories indicate relevance to the
endpoints, not relationship evidence by themselves. Do not transfer the current
user's or either endpoint's standalone traits, preferences, intentions, or actions
into a relationship claim. Mere co-occurrence is not relationship evidence. Look
for recurring interaction patterns, cooperation, friction, trust, initiative, and
meaningful changes in closeness, tone, or status. Recurring patterns require
multiple distinct source memories; a narrow inference from one highly diagnostic
interaction is allowed when calibrated to that evidence. Do not use projections,
inferred memories, or hypotheses as evidence. Distinguish current from historical
conditions. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""

CONTINUITY_SYSTEM_PROMPT = """Rebuild compact cross-session continuity.
Return only the supplied JSON schema. Keep ongoing threads, recent decisions,
pending actions, and context useful for the next conversation. Do not make
long-term personality inferences or copy person/user profiles; the current user
is not a Person. Use the language that best matches supplied messages and prior
continuity; keep IDs and enum values unchanged.
"""

USER_MODEL_REASONING_SYSTEM_PROMPT = """Reason carefully about the fixed current user from
active memories marked about_user. Capture preferences, communication preferences, goals, current
situations, and practical interaction guidance. Generalize recurring patterns
when supported, but do not turn a one-off event into a stable trait or include
another person's identity. Use valid_from and valid_to to separate current
conditions from historical events. Do not use projections, inferred memories, or
hypotheses as evidence. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""

QUERY_SYNTHESIS_SYSTEM_PROMPT = """Answer the read-only question using the supplied
social-memory context. Return concise plain text only (no JSON wrapper or code
fence). Use facts and supported inferences in the context to give a direct,
useful answer; distinguish uncertainty and current conditions from historical
events. Separate Person records are not evidence that they represent different
real people; when identities may overlap, state the ambiguity instead of asserting
a distinction. Do not invent facts or claim that anything was saved. Answer in
the language of the question.
"""


def schema_instruction(result_type: type[BaseModel]) -> str:
    """Return a compact instruction containing a Pydantic JSON schema."""

    schema = result_type.model_json_schema()
    return "Output schema (JSON Schema):\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )


def _json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


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
        "Person referenced by a memory in its `people` refs; express who said or "
        "did what in the memory content itself. Preserve reported claims as "
        "`reported`, not facts. Set `about_user` for a claim or event about "
        f"{user_name}; {user_name} must never appear in "
        "`people`. "
        "Record valid_from/valid_to when the message gives a time bound.\n\n"
        "Recent context (context only):\n"
        + _json(list(context))
        + "\n\nKnown people (identity hints only):\n"
        + _json(list(known_people))
        + "\nReuse a known person's canonical display_name when the messages refer "
        "to them. If the messages explicitly introduce a new short name, return "
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


def relationship_reasoning_prompt(
    relationship: RelationshipView,
    memories: list[MemoryView] | tuple[MemoryView, ...],
    user_name: str = "CurrentUser",
) -> str:
    """Build the user prompt for a relationship projection refresh."""

    return (
        f"The fixed current user is named {user_name!r}; the current user is not "
        "an endpoint Person. Rebuild the projection for the relationship between "
        "the two endpoint People in the input. Linked memories indicate relevance, "
        "not relationship evidence; summarize only interactions between the endpoints. "
        "Use only the supplied context; do not transfer standalone traits, preferences, "
        "or actions into a relationship claim.\n\nTarget relationship:\n"
        + _json(relationship)
        + "\n\nRelevant memories:\n"
        + _json(list(memories))
    )


def _evidence_lines(memories: list[MemoryView] | tuple[MemoryView, ...]) -> str:
    """Compact, injection-resistant enough-for-reading evidence representation."""
    return "\n".join(
        f"- id={m.id!r} kind={m.kind!r} basis={m.basis!r} derivation_sources={'unavailable' if m.basis == 'inferred' else 'n/a'} text={json.dumps(m.content, ensure_ascii=False)}"
        for m in memories
    ) or "- (none)"


def _hypothesis_lines(hypotheses: list[HypothesisView] | tuple[HypothesisView, ...]) -> str:
    return "\n".join(
        f"- id={h.id!r} confidence={h.confidence!r} evidence={[e.memory_id for e in h.evidence]!r} text={json.dumps(h.content, ensure_ascii=False)}"
        for h in hypotheses
    ) or "- (none)"


def owner_reasoning_prefix(
    target: PersonView | RelationshipView | UserModelView,
    evidence_memories: list[MemoryView] | tuple[MemoryView, ...],
    inferred_memories: list[MemoryView] | tuple[MemoryView, ...],
    hypotheses: list[HypothesisView] | tuple[HypothesisView, ...],
    *, user_name: str = "CurrentUser",
) -> str:
    """Shared immutable prefix for both stages of an owner reasoning pair."""
    return (
        f"<owner-reasoning user={json.dumps(user_name)}>\n<target>\n"
        + _json(target) + "\n</target>\n<evidence-memories>\n"
        + _evidence_lines(evidence_memories) + "\n</evidence-memories>\n"
        + "<current-inferred-memories comparison-only=\"true\">\n"
        + _evidence_lines(inferred_memories) + "\n</current-inferred-memories>\n"
        + "<open-hypotheses comparison-only=\"true\">\n" + _hypothesis_lines(hypotheses)
        + "\n</open-hypotheses>\nOnly evidence-memories are evidence. Current inferred memories and open hypotheses may be reviewed for duplication or explicit lifecycle actions, never used as support."
    )


def projection_stage_prompt() -> str:
    return "<stage>Return only the requested projection/card. Do not output inferred-memory or hypothesis actions.</stage>"


def actions_stage_prompt() -> str:
    return "<stage>Review the projection above. Return only explicit inferred-memory and hypothesis actions. Omission is always no-op. IDs must be from supplied context.</stage>"


def user_model_reasoning_prompt(
    memories: list[MemoryView], user_name: str = "CurrentUser"
) -> str:
    return (
        f"Rebuild the compact profile card for {user_name} from these active "
        "memories only. Keep it bounded and useful for interaction.\n\nActive "
        "about-user memories:\n" + _json(memories)
    )


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


def query_synthesis_prompt(question: str, context: QueryContext) -> str:
    """Build the user prompt for read-only query synthesis."""

    return (
        "Question:\n"
        + question
        + "\n\nContext (people, relationships, and memories):\n"
        + _json(context)
    )


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "QUERY_SYNTHESIS_SYSTEM_PROMPT",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
    "CONTINUITY_SYSTEM_PROMPT",
    "extraction_prompt",
    "person_reasoning_prompt",
    "query_synthesis_prompt",
    "relationship_reasoning_prompt",
    "owner_reasoning_prefix",
    "projection_stage_prompt",
    "actions_stage_prompt",
    "user_model_reasoning_prompt",
    "continuity_prompt",
    "schema_instruction",
]
