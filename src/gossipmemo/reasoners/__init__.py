"""Reasoners: one module per induction pass, each owning its own load/call/apply logic.

`base` holds the shared `Reasoner` seam. Extraction and query synthesis are
not reasoners in this sense -- extraction stays outside the abstraction
entirely (see `SocialMemoryWorld._extract`), and query synthesis is a single
read-only call with no watermark to commit.
"""

from __future__ import annotations

from .base import DescriptorReasoner, Reasoner
from .continuity import build_continuity_reasoner
from .coverage import build_coverage_reasoner
from .learning_goals import build_learning_goals_reasoner
from .person import PersonReasoner
from .relationship import RelationshipReasoner
from .user_model import build_user_model_reasoner

__all__ = [
    "DescriptorReasoner",
    "PersonReasoner",
    "Reasoner",
    "RelationshipReasoner",
    "build_continuity_reasoner",
    "build_coverage_reasoner",
    "build_learning_goals_reasoner",
    "build_user_model_reasoner",
]
