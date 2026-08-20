"""Reasoners: one module per induction pass, each owning its own prompt(s) and
load/call/apply logic.

`base` holds the shared `Reasoner` seam. Every module in this package builds
a `Reasoner` the same way, via a `build_*_reasoner(store, model, settings)`
factory.
Query synthesis (`gossipmemo.query`) is not a reasoner -- it is a single
read-only call with no watermark to commit -- so it lives outside this
package.
"""

from __future__ import annotations

from .base import AttemptLoop, DescriptorReasoner, Reasoner
from .continuity import CONTINUITY_SYSTEM_PROMPT, build_continuity_reasoner, continuity_prompt
from .coverage import COVERAGE_AUDIT_SYSTEM_PROMPT, build_coverage_reasoner, coverage_audit_prompt
from .extraction import EXTRACTION_SYSTEM_PROMPT, build_extraction_reasoner, extraction_prompt
from .learning_goals import (
    GOAL_PLANNING_SYSTEM_PROMPT,
    build_learning_goals_reasoner,
    goal_candidate_prompt,
    goal_candidate_reduction_prompt,
    goal_reconciliation_prompt,
)
from .person import PERSON_REASONING_SYSTEM_PROMPT, build_person_reasoner
from .relationship import (
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    build_relationship_reasoner,
)
from .settings import ReasoningSettings
from .user_model import (
    USER_MODEL_REASONING_SYSTEM_PROMPT,
    build_user_model_reasoner,
)

__all__ = [
    "AttemptLoop",
    "CONTINUITY_SYSTEM_PROMPT",
    "COVERAGE_AUDIT_SYSTEM_PROMPT",
    "DescriptorReasoner",
    "EXTRACTION_SYSTEM_PROMPT",
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "ReasoningSettings",
    "Reasoner",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
    "build_continuity_reasoner",
    "build_coverage_reasoner",
    "build_extraction_reasoner",
    "build_learning_goals_reasoner",
    "build_person_reasoner",
    "build_relationship_reasoner",
    "build_user_model_reasoner",
    "continuity_prompt",
    "coverage_audit_prompt",
    "extraction_prompt",
    "goal_candidate_prompt",
    "goal_candidate_reduction_prompt",
    "goal_reconciliation_prompt",
]
