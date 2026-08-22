"""Prompt rendering: the helpers that turn live data into prompt text.

The wording itself lives in `prompts.defaults` and reaches these builders
through a `PromptLibrary`; what stays here is code -- the JSON-schema
instruction helper, `fill` for the fragments that interpolate a value, compact
evidence/hypothesis rendering, the owner-reasoning family shared by
person/relationship/user_model, and the two stage fragments the owner pair
reads from the library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from string import Template
from typing import Any

from pydantic import BaseModel

from ..models import (
    HypothesisView,
    MemoryView,
)
from .library import PromptLibrary


def fill(fragment: str, /, **values: str) -> str:
    """Substitute a fragment's `$name` placeholders.

    `Template` rather than `str.format`: prompt text is full of literal braces
    (JSON schemas, `{}` examples), which brace formatting would try to read as
    fields. `substitute` raises on a missing name instead of leaving it in the
    text; `PromptLibrary` has already checked at load time that an override
    uses exactly the names its use site passes.
    """

    return Template(fragment).substitute(values)


def schema_instruction(result_type: type[BaseModel]) -> str:
    """Return a compact instruction containing a Pydantic JSON schema."""

    schema = result_type.model_json_schema()
    return "Output schema (JSON Schema):\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )


def _plain(value: Any) -> Any:
    """Turn nested Pydantic models into JSON-serializable data.

    Recursion matters for containers: a list or dict of models must become a
    list or dict of JSON objects, not `repr` strings from
    `json.dumps(default=str)`.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, indent=2, default=str)


def _evidence_lines(memories: Sequence[Any]) -> str:
    """Compact, injection-resistant enough-for-reading evidence representation."""
    return "\n".join(
        f"- id={m.id!r} kind={m.kind!r} basis={m.basis!r} "
        f"text={json.dumps(m.content, ensure_ascii=False)}"
        for m in memories
    ) or "- (none)"


def _invalidated_lines(memories: Sequence[Any]) -> str:
    """Withdrawn records: what the card must stop asserting, and why."""
    return "\n".join(
        f"- id={m.id!r} status={m.status!r} kind={m.kind!r} "
        f"text={json.dumps(m.content, ensure_ascii=False)}"
        for m in memories
    ) or "- (none)"


def _hypothesis_lines(hypotheses: list[HypothesisView] | tuple[HypothesisView, ...]) -> str:
    return "\n".join(
        f"- id={h.id!r} confidence={h.confidence!r} "
        f"evidence={[e.memory_id for e in h.evidence]!r} "
        f"text={json.dumps(h.content, ensure_ascii=False)}"
        for h in hypotheses
    ) or "- (none)"


def owner_reasoning_prefix(
    target: BaseModel,
    evidence_memories: Sequence[Any],
    invalidated_memories: Sequence[Any],
    inferred_memories: list[MemoryView] | tuple[MemoryView, ...],
    hypotheses: list[HypothesisView] | tuple[HypothesisView, ...],
    *, prompts: PromptLibrary, user_name: str = "CurrentUser",
) -> str:
    """Shared immutable prefix for both stages of an owner reasoning pair.

    `target` carries the card this batch folds into, so the four rendered
    sections are four different kinds of input: the card so far, evidence,
    records withdrawn since the card was written, and comparison-only
    state. Invalidation gets its own section rather than a flag on an
    evidence line because it is neither evidence nor comparison -- it is a
    negative instruction, and `owner_evidence_scope_rule` draws its
    boundary by section.
    """
    return (
        f"<owner-reasoning user={json.dumps(user_name)}>\n<target>\n"
        + _json(target) + "\n</target>\n<evidence-memories>\n"
        + _evidence_lines(evidence_memories) + "\n</evidence-memories>\n"
        + "<invalidated-memories evidence=\"false\">\n"
        + _invalidated_lines(invalidated_memories) + "\n</invalidated-memories>\n"
        + "<current-inferred-memories comparison-only=\"true\">\n"
        + _evidence_lines(inferred_memories) + "\n</current-inferred-memories>\n"
        + "<open-hypotheses comparison-only=\"true\">\n" + _hypothesis_lines(hypotheses)
        + "\n</open-hypotheses>\n" + prompts.owner_fold_rule
        + "\n" + prompts.owner_invalidated_scope_rule
        + "\n" + prompts.owner_evidence_scope_rule
    )


def projection_stage_prompt(prompts: PromptLibrary) -> str:
    return prompts.projection_stage


def actions_stage_prompt(prompts: PromptLibrary) -> str:
    return prompts.actions_stage


__all__ = [
    "actions_stage_prompt",
    "fill",
    "owner_reasoning_prefix",
    "projection_stage_prompt",
    "schema_instruction",
]
