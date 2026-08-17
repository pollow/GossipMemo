"""Coverage-audit reasoner and its prompt.

Coverage is one recursive table of entries: an entry summarizes how well one
path under one root is understood, and the root-level entry (empty path) is
the overview of that root. Auditing fans out over roots -- one request per
root, carrying that root's active entries plus a chunk of its new evidence --
so an entry's root is decided by which request produced it and never by a
field the model fills in.

Each attempt audits exactly one budget-sized chunk of one root and commits
it against that root's own cursor, so entries written by an earlier chunk are
visible to the next one and a failure never rolls back the roots (or chunks)
that already landed.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from ..chunking import greedy_chunks
from ..models import COVERAGE_CRITERIA, CoverageEntryView, ExtractedCoverageAudit, MemoryView
from ..priority import current_call_label, current_call_tier
from ..prompts import COVERAGE_ROOT_VIEWPOINTS, _evidence_lines, schema_instruction
from ..store import WorldStore
from ..transport import ChatCompletionRequest, ChatMessage, LlmTransport, structured
from .base import DescriptorReasoner

COVERAGE_AUDIT_SYSTEM_PROMPT = """Summarize how well one area of a person's life and
persona is already understood. Return only the supplied JSON schema. An entry is a
summary over many memories -- roughly dozens of memories into a short paragraph -- not a
retelling of them and not memoir prose. Write only what is known; never write what is
missing, unclear, or worth asking about. Do not invent facts, evidence, or private
details, and do not diagnose. Keep entries concise and in the language of the evidence."""


def coverage_audit_prompt(
    root: str, entries: Sequence[CoverageEntryView], memories: Sequence[MemoryView],
) -> str:
    """One root's current entries plus one bounded chunk of its new evidence."""
    entry_lines = "\n".join(
        f"- id={item.id!r} path={item.path!r} content={item.content!r}" for item in entries
    ) or "- (none)"
    return (
        f"<coverage-root id={root!r} facet={COVERAGE_CRITERIA.get(root, '')!r}>\n"
        + COVERAGE_ROOT_VIEWPOINTS.get(root, "") + "\n</coverage-root>\n"
        "<current-entries>\n" + entry_lines + "\n</current-entries>\n"
        "<new-evidence>\n" + _evidence_lines(list(memories)) + "\n</new-evidence>\n"
        "Fold this evidence into the entries for this root. Add an entry for a topic that "
        "the entries do not cover yet, and modify an entry whose summary this evidence "
        "changes or extends; leaving an entry out changes nothing. An entry that only one "
        "memory supports is almost always wrong -- that is a single event, not an "
        "understanding of a topic. Keep exactly one entry with an empty path: the overview "
        "of this root, naming which areas exist under it and how much is understood about "
        "each. Paths are free text; reuse a stored path when you mean the same area, and do "
        "not renumber or normalize the others. Keep content under about two hundred words: "
        "when an entry outgrows that, split it by narrowing that entry and adding the "
        "areas it no longer covers. To merge two entries, rewrite one to absorb the other "
        "and modify the other with status \"superseded\"."
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


async def _audit_coverage(
    transport: "LlmTransport", root: str, entries: Sequence[CoverageEntryView],
    memories: Sequence[MemoryView],
) -> tuple[ExtractedCoverageAudit, list[MemoryView]]:
    """Audit the largest prefix of `memories` that fits one request.

    Returns the audit together with the evidence it actually read, since the
    caller advances this root's cursor by exactly that much. A root whose
    entries alone no longer fit the budget raises rather than dropping any of
    them: compacting such a root is its own pass, not a silent truncation.
    """
    context_budget = transport.context_budget

    def request_for(chunk: Sequence[MemoryView]) -> ChatCompletionRequest:
        return _structured_request(
            transport, COVERAGE_AUDIT_SYSTEM_PROMPT,
            coverage_audit_prompt(root, entries, chunk), ExtractedCoverageAudit,
        )

    def fits(chunk: Sequence[MemoryView]) -> bool:
        return context_budget.report(context_budget.estimate_request(request_for(chunk))).fits

    def check(chunk: Sequence[MemoryView]) -> None:
        context_budget.check(context_budget.estimate_request(request_for(chunk)))

    chunk = greedy_chunks(list(memories), fits, check)[0]
    request = request_for(chunk)
    _, result = await structured(
        transport, request.messages, ExtractedCoverageAudit,
        tier=current_call_tier(), label=current_call_label(),
    )
    return result, chunk


def build_coverage_reasoner(store: WorldStore, model: "LlmTransport") -> DescriptorReasoner:
    """Audit one root's next evidence chunk per attempt, until none is behind."""

    audit_coverage = partial(_audit_coverage, model)

    def load_context(space_id: str):
        return store.coverage_context(space_id)

    def call(space_id: str, context):
        root, entries, memories = context
        return "audit-coverage", audit_coverage, (root.root, entries, memories)

    def apply(space_id: str, context, result) -> bool:
        root, entries, _ = context
        audit, audited = result
        return store.apply_coverage_audit(
            space_id, root.root, root.source_watermark, root.source_cursor_id, audit,
            {memory.id for memory in audited}, {entry.id for entry in entries},
        )

    def continue_when(context, result, applied: bool) -> bool:
        # Both outcomes reload: another chunk or root is usually still
        # behind, and a lost CAS means the fresh cursor is worth re-reading.
        return True

    return DescriptorReasoner("coverage", load_context, call, apply, continue_when)


__all__ = ["COVERAGE_AUDIT_SYSTEM_PROMPT", "build_coverage_reasoner", "coverage_audit_prompt"]
