"""Coverage-map audit reasoner."""

from __future__ import annotations

from ..llm import LlmModel
from ..queue import ReasonerCallQueue
from ..store import WorldStore
from .base import DescriptorReasoner


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


__all__ = ["build_coverage_reasoner"]
