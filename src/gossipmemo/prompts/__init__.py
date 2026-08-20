"""Prompt scaffolding shared across reasoners.

`defaults` holds the shipped wording as plain constants, `library` wraps it in
the overridable `PromptLibrary` that `ReasoningSettings` carries, and `render`
holds the code that turns live data into prompt text. This module is the façade:
importing a fragment, the library, or a rendering helper from `gossipmemo.prompts`
works regardless of which of the three it lives in.
"""

from __future__ import annotations

from .defaults import (
    ACTIONS_STAGE_PROMPT,
    CONTINUITY_SYSTEM_PROMPT,
    COVERAGE_AUDIT_SYSTEM_PROMPT,
    COVERAGE_METHOD,
    COVERAGE_ROOT_BLIND_SPOTS,
    COVERAGE_ROOT_VIEWPOINTS,
    EXTRACTION_SYSTEM_PROMPT,
    GOAL_PLANNING_SYSTEM_PROMPT,
    PERSON_REASONING_SYSTEM_PROMPT,
    PROJECTION_STAGE_PROMPT,
    QUERY_SYNTHESIS_SYSTEM_PROMPT,
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    USER_MODEL_REASONING_SYSTEM_PROMPT,
)
from .library import PromptLibrary
from .render import (
    actions_stage_prompt,
    owner_evidence_digest_prompt,
    owner_reasoning_prefix,
    projection_stage_prompt,
    schema_instruction,
)

__all__ = [
    "ACTIONS_STAGE_PROMPT",
    "CONTINUITY_SYSTEM_PROMPT",
    "COVERAGE_AUDIT_SYSTEM_PROMPT",
    "COVERAGE_METHOD",
    "COVERAGE_ROOT_BLIND_SPOTS",
    "COVERAGE_ROOT_VIEWPOINTS",
    "EXTRACTION_SYSTEM_PROMPT",
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "PROJECTION_STAGE_PROMPT",
    "QUERY_SYNTHESIS_SYSTEM_PROMPT",
    "PromptLibrary",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
    "actions_stage_prompt",
    "owner_evidence_digest_prompt",
    "owner_reasoning_prefix",
    "projection_stage_prompt",
    "schema_instruction",
]
