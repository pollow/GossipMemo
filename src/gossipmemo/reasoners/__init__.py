"""Reasoners: one module per induction pass, each owning its own prompt(s) and
load/call/apply logic.

`base` holds the shared `Reasoner` seam. Extraction (`extraction.py`) and
query synthesis (`query.py`) are not reasoners in that sense -- extraction
stays outside the abstraction entirely (see `SocialMemoryWorld._extract`),
and query synthesis is a single read-only call with no watermark to commit
-- but their prompts still live alongside the rest of their handling.
"""

from __future__ import annotations

from .base import DescriptorReasoner, Reasoner
from .continuity import CONTINUITY_SYSTEM_PROMPT, build_continuity_reasoner, continuity_prompt
from .coverage import COVERAGE_AUDIT_SYSTEM_PROMPT, build_coverage_reasoner, coverage_audit_prompt
from .extraction import EXTRACTION_SYSTEM_PROMPT, extraction_prompt
from .learning_goals import (
    GOAL_PLANNING_SYSTEM_PROMPT,
    build_learning_goals_reasoner,
    goal_candidate_prompt,
    goal_candidate_reduction_prompt,
    goal_planning_prompt,
    goal_reconciliation_prompt,
)
from .person import PERSON_REASONING_SYSTEM_PROMPT, PersonReasoner, person_reasoning_prompt
from .query import QUERY_SYNTHESIS_SYSTEM_PROMPT, query_synthesis_prompt
from .relationship import (
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    RelationshipReasoner,
    relationship_reasoning_prompt,
)
from .user_model import (
    USER_MODEL_REASONING_SYSTEM_PROMPT,
    build_user_model_reasoner,
    user_model_reasoning_prompt,
)

__all__ = [
    "CONTINUITY_SYSTEM_PROMPT",
    "COVERAGE_AUDIT_SYSTEM_PROMPT",
    "DescriptorReasoner",
    "EXTRACTION_SYSTEM_PROMPT",
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "PersonReasoner",
    "QUERY_SYNTHESIS_SYSTEM_PROMPT",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "Reasoner",
    "RelationshipReasoner",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
    "build_continuity_reasoner",
    "build_coverage_reasoner",
    "build_learning_goals_reasoner",
    "build_user_model_reasoner",
    "continuity_prompt",
    "coverage_audit_prompt",
    "extraction_prompt",
    "goal_candidate_prompt",
    "goal_candidate_reduction_prompt",
    "goal_planning_prompt",
    "goal_reconciliation_prompt",
    "person_reasoning_prompt",
    "query_synthesis_prompt",
    "relationship_reasoning_prompt",
    "user_model_reasoning_prompt",
]
