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
    QueryContext,
    RelationshipReasoningResult,
    RelationshipView,
)


EXTRACTION_SYSTEM_PROMPT = """You extract provenance-aware social memory from a batch of messages.
Return only a JSON object matching the supplied schema. Do not add Markdown or
explanatory prose. Keep the original meaning and uncertainty. Create people
only when the text provides a usable reference; use person refs that can be
resolved from display names or aliases. A memory basis must reflect how the
message supports the claim (stated, observed, reported, inferred, or manual).
Do not treat a mere co-occurrence of two people as a relationship.
"""

PERSON_REASONING_SYSTEM_PROMPT = """You maintain a person's durable profile from
active, provenance-bearing memories. Return only a JSON object matching the
supplied schema. Profile-card text must distinguish direct, reported, and
inferred information and should not invent evidence. Inferred memories are
optional; each one must cite one or more supplied source memory IDs and must
not use a profile card as evidence.
"""

RELATIONSHIP_REASONING_SYSTEM_PROMPT = """You maintain a relationship projection
from active, provenance-bearing memories. Return only a JSON object matching
the supplied schema. Preserve uncertainty and do not infer a relationship
from people merely appearing in the same message. Inferred memories are
optional; each one must cite one or more supplied source memory IDs.
"""

USER_MODEL_REASONING_SYSTEM_PROMPT = """You maintain a compact, bounded profile
of the current user from active memories explicitly marked about_user. Rebuild
the profile from the supplied memories; do not append, invent, or include a
Person identity. Return only a JSON object matching the supplied schema.
"""

QUERY_SYNTHESIS_SYSTEM_PROMPT = """You answer a read-only question using the supplied
social-memory context. Return concise plain text only (no JSON wrapper and no
Markdown code fence). Distinguish evidence from inference, mention relevant
uncertainty, and do not claim facts absent from the context. The context is
read-only: do not suggest that your answer has been saved.
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


def extraction_prompt(messages: list[ModelMessage]) -> str:
    """Build the user prompt for :class:`~models.ExtractionResult`."""

    policies = {message.extraction_policy for message in messages}
    if "comprehensive" in policies:
        policy_name = "comprehensive"
    elif "balanced" in policies:
        policy_name = "balanced"
    else:
        policy_name = "conservative"
    policy = {
        "conservative": (
            "Retain only explicit, durable facts, events, preferences, plans, "
            "and situations. Skip weak implications and conversational filler."
        ),
        "balanced": (
            "Retain explicit durable information and carefully supported social "
            "signals; skip transient filler and speculative personality labels."
        ),
        "comprehensive": (
            "Retain explicit information plus useful weak social signals as "
            "impression/inferred memories, preserving uncertainty and evidence."
        ),
    }[policy_name]
    return (
        f"Extraction policy: {policy_name}. {policy}\n"
        "Extract the messages together as one conversational context.\n"
        "The user/assistant author role is context and never a Person. List every "
        "Person referenced by a memory in its `people` refs; express who said or "
        "did what in the memory content itself. Preserve reported claims as "
        "`reported`, not facts. Set `about_user` only when the memory is a durable "
        "fact, preference, plan, situation, or other claim about the current user; "
        "the current user is not a Person and must never appear in `people`.\n\n"
        "Messages:\n"
        + _json(messages)
    )


def person_reasoning_prompt(
    person: PersonView, memories: list[MemoryView] | tuple[MemoryView, ...]
) -> str:
    """Build the user prompt for a person projection refresh."""

    return (
        "Rebuild the profile card for this person from the active memories. "
        "Use only the supplied context.\n\nPerson:\n"
        + _json(person)
        + "\n\nActive memories:\n"
        + _json(list(memories))
    )


def relationship_reasoning_prompt(
    relationship: RelationshipView,
    memories: list[MemoryView] | tuple[MemoryView, ...],
) -> str:
    """Build the user prompt for a relationship projection refresh."""

    return (
        "Rebuild the relationship projection from the active memories. "
        "Use only the supplied context.\n\nRelationship:\n"
        + _json(relationship)
        + "\n\nRelevant memories:\n"
        + _json(list(memories))
    )


def user_model_reasoning_prompt(memories: list[MemoryView]) -> str:
    return (
        "Rebuild the compact profile card for the current user from these active "
        "memories only.\n\nActive about-user memories:\n" + _json(memories)
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
    "extraction_prompt",
    "person_reasoning_prompt",
    "query_synthesis_prompt",
    "relationship_reasoning_prompt",
    "user_model_reasoning_prompt",
    "schema_instruction",
]
