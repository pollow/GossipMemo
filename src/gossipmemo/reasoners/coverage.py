"""Coverage-map audit reasoner and its prompt.

Coverage is an accumulator, not a snapshot (unlike the owner family), so
an oversized audit is bounded by paginating evidence across several
requests rather than lossily digesting it: `_audit_coverage` packs
memories into `chunking.greedy_chunks`-sized requests and applies each
chunk's patch independently, filtering forged evidence IDs per chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from ..chunking import greedy_chunks
from ..models import CoverageAuditPatch, CoverageMapView, HypothesisView, MemoryView
from ..priority import current_call_label, current_call_tier
from ..prompts import (
    COVERAGE_METHOD,
    COVERAGE_RUBRIC,
    _evidence_lines,
    _hypothesis_lines,
    _json,
    schema_instruction,
)
from ..store import WorldStore
from ..transport import ChatCompletionRequest, ChatMessage, LlmTransport, structured
from .base import DescriptorReasoner

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


def _structured_request(
    transport: "LlmTransport", system_prompt: str, user_prompt: str, result_type: type,
) -> ChatCompletionRequest:
    return transport.prepare(
        [
            ChatMessage(role="system", content=system_prompt +
                        "\n\n" + schema_instruction(result_type)),
            ChatMessage(role="user", content=user_prompt),
        ],
        structured=True,
    )


def _coverage_skeleton(coverage: CoverageMapView) -> CoverageMapView:
    """Bound coverage prose/evidence IDs for scaffolding-only sizing.

    Shared with `learning_goals._bounded_goal_context`, since both need the
    same minimum-viable coverage map before spending remaining budget on
    comparison-only hypotheses or goals.
    """
    criteria = {
        identifier: {
            "level": value.get("level", "unknown"),
            "known_state": str(value.get("known_state", ""))[:800],
            "evidence_memory_ids": list(value.get("evidence_memory_ids", []))[:32],
        }
        for identifier, value in coverage.criteria.items()
    }
    boundaries = [
        item.model_copy(update={
            "summary": item.summary[:600],
            "evidence_memory_ids": item.evidence_memory_ids[:32],
        })
        for item in coverage.boundaries
    ]
    return coverage.model_copy(update={
        "criteria": criteria,
        "boundaries": boundaries,
        "life_periods": [str(item)[:600] for item in coverage.life_periods[:30]],
        "relationship_arcs": [str(item)[:600] for item in coverage.relationship_arcs[:30]],
        "behavioral_contexts": [str(item)[:600] for item in coverage.behavioral_contexts[:30]],
    })


def _bounded_coverage_context(
    transport: "LlmTransport", coverage: CoverageMapView, hypotheses: Sequence[HypothesisView],
) -> tuple[CoverageMapView, list[HypothesisView]]:
    # Bound comparison-only hypotheses and boundary prose before evidence is
    # added. IDs remain present whenever scaffold space permits it.
    context_budget = transport.context_budget

    def fits(candidate_coverage: CoverageMapView, candidate_hypotheses: Sequence[HypothesisView]) -> bool:
        request = _structured_request(
            transport, COVERAGE_AUDIT_SYSTEM_PROMPT,
            coverage_audit_prompt(candidate_coverage, [], list(candidate_hypotheses)),
            CoverageAuditPatch,
        )
        return context_budget.report(context_budget.estimate_request(request)).fits

    if fits(coverage, hypotheses):
        return coverage, list(hypotheses)
    skeleton_hypotheses = [
        item.model_copy(update={"content": "", "evidence": []}) for item in hypotheses
    ]
    if fits(coverage, skeleton_hypotheses):
        return coverage, skeleton_hypotheses

    bounded = _coverage_skeleton(coverage)
    if not fits(bounded, []):
        raise ValueError("coverage prompt scaffolding exceeds context budget")
    chosen: list[HypothesisView] = []
    for hypothesis in hypotheses:
        skeleton = hypothesis.model_copy(update={"content": "", "evidence": []})
        if fits(bounded, [*chosen, skeleton]):
            chosen.append(skeleton)
    return bounded, chosen


def _filter_coverage_patch(patch: CoverageAuditPatch, evidence_ids: set[str]) -> CoverageAuditPatch:
    return CoverageAuditPatch(
        criteria=[
            item.model_copy(update={
                "evidence_memory_ids": [id_ for id_ in item.evidence_memory_ids if id_ in evidence_ids],
            })
            for item in patch.criteria
        ],
        boundary_upserts=[
            item.model_copy(update={
                "evidence_memory_ids": [id_ for id_ in item.evidence_memory_ids if id_ in evidence_ids],
            })
            for item in patch.boundary_upserts
        ],
        boundary_transitions=patch.boundary_transitions,
        life_periods=patch.life_periods,
        relationship_arcs=patch.relationship_arcs,
        behavioral_contexts=patch.behavioral_contexts,
    )


async def _audit_coverage(
    transport: "LlmTransport", coverage: CoverageMapView, memories: Sequence[MemoryView],
    hypotheses: Sequence[HypothesisView] = (),
) -> CoverageAuditPatch:
    context_budget = transport.context_budget
    normal = _structured_request(
        transport, COVERAGE_AUDIT_SYSTEM_PROMPT,
        coverage_audit_prompt(coverage, list(memories), list(hypotheses)), CoverageAuditPatch,
    )
    if context_budget.report(context_budget.estimate_request(normal)).fits:
        _, result = await structured(
            transport, normal.messages, CoverageAuditPatch,
            tier=current_call_tier(), label=current_call_label(),
        )
        return result

    bounded_coverage, bounded_hypotheses = _bounded_coverage_context(
        transport, coverage, hypotheses)

    def request_for(chunk: Sequence[MemoryView]) -> ChatCompletionRequest:
        return _structured_request(
            transport, COVERAGE_AUDIT_SYSTEM_PROMPT,
            coverage_audit_prompt(bounded_coverage, list(
                chunk), bounded_hypotheses), CoverageAuditPatch,
        )

    def fits(chunk: Sequence[MemoryView]) -> bool:
        return context_budget.report(context_budget.estimate_request(request_for(chunk))).fits

    def check(chunk: Sequence[MemoryView]) -> None:
        context_budget.check(context_budget.estimate_request(request_for(chunk)))

    patches: list[CoverageAuditPatch] = []
    for chunk in greedy_chunks(list(memories), fits, check):
        request = request_for(chunk)
        _, result = await structured(
            transport, request.messages, CoverageAuditPatch,
            tier=current_call_tier(), label=current_call_label(),
        )
        patches.append(_filter_coverage_patch(result, {memory.id for memory in chunk}))

    return CoverageAuditPatch(
        criteria=[item for patch in patches for item in patch.criteria],
        boundary_upserts=[item for patch in patches for item in patch.boundary_upserts],
        boundary_transitions=[item for patch in patches for item in patch.boundary_transitions],
        life_periods=[item for patch in patches for item in patch.life_periods],
        relationship_arcs=[item for patch in patches for item in patch.relationship_arcs],
        behavioral_contexts=[item for patch in patches for item in patch.behavioral_contexts],
    )


def build_coverage_reasoner(store: WorldStore, model: "LlmTransport") -> DescriptorReasoner:
    """Audit all bounded chunks before a single goal-planning pass."""

    audit_coverage = partial(_audit_coverage, model)

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
        return "audit-coverage", audit_coverage, (coverage, memories, hypotheses)

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

    return DescriptorReasoner("coverage", load_context, call, apply, continue_when)


__all__ = ["COVERAGE_AUDIT_SYSTEM_PROMPT", "build_coverage_reasoner", "coverage_audit_prompt"]
