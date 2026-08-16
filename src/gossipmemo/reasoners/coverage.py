"""Coverage-map audit reasoner and its prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import CoverageMapView, HypothesisView, MemoryView
from ..prompts import COVERAGE_METHOD, COVERAGE_RUBRIC, _evidence_lines, _hypothesis_lines, _json
from ..queue import ReasonerCallQueue
from ..store import WorldStore
from .base import DescriptorReasoner

if TYPE_CHECKING:
    from ..llm import LlmModel

COVERAGE_AUDIT_SYSTEM_PROMPT = """Audit long-term autobiographical and persona coverage.
Return only the supplied JSON schema. Coverage is a summary of supported evidence,
not a profile and not an invitation to disclose. A hypothesis may identify an edge,
blind spot, or conflict, but never raises a coverage level. Preserve uncertainty:
unknown, private, or deferred material is valid and must not be treated as a gap to
press. Do not diagnose pathology. Use the supplied memory IDs exactly; do not invent
facts, evidence, or private details. Keep natural-language summaries concise and in
the language of the evidence."""


def coverage_audit_prompt(
    coverage: CoverageMapView, memories: list[MemoryView], hypotheses: list[HypothesisView]
) -> str:
    """Immutable audit prefix plus one bounded evidence chunk."""
    return (
        "<coverage-rubric>\n" + COVERAGE_RUBRIC + "\n" + COVERAGE_METHOD + "\n</coverage-rubric>\n"
        "<current-coverage-map>\n" + _json(coverage) + "\n</current-coverage-map>\n"
        "<new-evidence>\n" + _evidence_lines(memories) + "\n</new-evidence>\n"
        "<open-hypotheses comparison-only=\"true\">\n" + _hypothesis_lines(hypotheses)
        + "\n</open-hypotheses>\nApply a patch only for this chunk. Preserve prior coverage unless new evidence changes it. "
        "Hypotheses may add a boundary or conflict with hypothesis_id, never evidence; a hypothesis never raises a coverage level. "
        "Each criterion patch needs only its stable parent criterion_id. Keep inventories compact and additive."
    )


def build_coverage_reasoner(store: WorldStore, model: LlmModel, queue: ReasonerCallQueue) -> DescriptorReasoner:
    """Audit all bounded chunks before a single goal-planning pass."""

    def load_context(space_id: str):
        context = store.coverage_context(space_id, limit=None)
        if not context:
            return None
        _, memories, _, _ = context
        if not memories:
            return None
        return context

    def call(space_id: str, context):
        coverage, memories, hypotheses, _ = context
        return "audit-coverage", model.audit_coverage, (coverage, memories, hypotheses)

    def apply(space_id: str, context, result) -> bool:
        coverage, memories, hypotheses, _ = context
        return store.apply_coverage_audit(
            space_id, coverage.source_watermark, coverage.source_cursor_id, result,
            {memory.id for memory in memories}, {boundary.id for boundary in coverage.boundaries},
            {hypothesis.id for hypothesis in hypotheses},
        )

    def continue_when(context, result, applied: bool) -> bool:
        if not applied:
            return True
        _, _, _, pending = context
        return pending

    return DescriptorReasoner("coverage", queue, load_context, call, apply, continue_when)


__all__ = ["COVERAGE_AUDIT_SYSTEM_PROMPT", "build_coverage_reasoner", "coverage_audit_prompt"]
