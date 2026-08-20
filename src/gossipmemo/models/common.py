from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


MemoryBasis = Literal["stated", "observed", "reported", "inferred", "manual"]
MemoryKind = Literal["fact", "event", "preference", "plan", "situation", "impression"]
HypothesisOwnerKind = Literal["user", "person", "relationship"]
HypothesisStatus = Literal["open", "promoted", "rejected", "superseded", "retired"]
HypothesisConfidence = Literal["low", "medium", "high"]
HypothesisEvidenceRole = Literal["support", "counter"]
CoverageEntryStatus = Literal["active", "superseded"]
LearningGoalStatus = Literal["open", "partial", "answered", "deferred", "retired"]

# These are intentionally prompt-native IDs rather than a normalized rubric
# table. They are the coverage *roots*: one audit request per root, so a root
# is decided by which request produced an entry and never by a field the model
# fills in. Everything below a root is free-text `path`.
COVERAGE_CRITERIA: dict[str, str] = {
    "M1": "life_chapters", "M2": "everyday_life", "M3": "turning_points",
    "M4": "people_and_relationship_arcs", "M5": "places_and_context",
    "M6": "lived_scenes", "M7": "inner_experience", "M8": "themes_and_change",
    "M9": "unresolved_threads", "P1": "identity_and_self_story",
    "P2": "values_and_tradeoffs", "P3": "worldview_and_beliefs",
    "P4": "goals_motives_fears", "P5": "reasoning_and_decisions",
    "P6": "emotional_patterns", "P7": "social_style_and_boundaries",
    "P8": "preferences_and_routines", "P9": "voice_and_expression",
    "P10": "context_and_exceptions", "P11": "skills_and_knowledge",
}


COVERAGE_ROOTS: tuple[str, ...] = tuple(COVERAGE_CRITERIA)


class HypothesisEvidence(BaseModel):
    memory_id: str = Field(min_length=1)
    role: HypothesisEvidenceRole = "support"
