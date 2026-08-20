"""Reasoners: one module per induction pass, each owning its load/call/apply
logic and the prompt builders that shape its requests.

Static prompt text is not here: it lives in `gossipmemo.prompts.defaults` and
reaches a reasoner as `ReasoningSettings.prompts`, a `PromptLibrary` a
deployment may override field by field. A reasoner module holds the code that
assembles a request around that text.

`base` holds the shared `Reasoner` seam. Every module in this package builds
a `Reasoner` the same way, via a `build_*_reasoner(store, model, settings)`
factory.
Query synthesis (`gossipmemo.query`) is not a reasoner -- it is a single
read-only call with no watermark to commit -- so it lives outside this
package.
"""

from __future__ import annotations

from .base import AttemptLoop, DescriptorReasoner, Reasoner
from .continuity import build_continuity_reasoner, continuity_prompt
from .coverage import build_coverage_reasoner, coverage_audit_prompt
from .extraction import build_extraction_reasoner, extraction_prompt
from .learning_goals import (
    build_learning_goals_reasoner,
    goal_candidate_prompt,
    goal_candidate_reduction_prompt,
    goal_reconciliation_prompt,
)
from .person import build_person_reasoner
from .relationship import build_relationship_reasoner
from .settings import DEFAULT_PROMPTS, ReasoningSettings
from .user_model import build_user_model_reasoner

__all__ = [
    "AttemptLoop",
    "DEFAULT_PROMPTS",
    "DescriptorReasoner",
    "ReasoningSettings",
    "Reasoner",
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
