"""Coverage-audit reasoner and its prompt.

Coverage is one recursive table of entries: an entry summarizes what is known
on one path under one root, and the root-level entry (empty path) is the
overview of that root. Auditing fans out over roots -- one request per
root, carrying that root's active entries plus a chunk of its new evidence --
so an entry's root is decided by which request produced it and never by a
field the model fills in.

Each attempt audits exactly one budget-sized chunk of one root and commits
it against that root's own cursor, so entries written by an earlier chunk are
visible to the next one and a failure never rolls back the roots (or chunks)
that already landed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

from ..chunking import greedy_chunks
from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient
from ..models import COVERAGE_CRITERIA, CoverageEntryView, ExtractedCoverageAudit, MemoryView
from ..priority import current_call_label, current_call_tier
from ..prompts import PromptLibrary, schema_instruction
from ..prompts.render import _evidence_lines
from ..store import WorldStore
from ..transport import ChatCompletionRequest, ChatMessage, LlmTransport, structured
from .base import DescriptorReasoner
from .dedup_priority import similarity_priority_order
from .settings import ReasoningSettings


def coverage_audit_prompt(
    root: str, entries: Sequence[CoverageEntryView], memories: Sequence[MemoryView],
    *, prompts: PromptLibrary,
) -> str:
    """One root's current entries plus one bounded chunk of its new evidence."""
    entry_lines = "\n".join(
        f"- id={item.id!r} path={item.path!r} content={item.content!r}" for item in entries
    ) or "- (none)"
    return (
        f"<coverage-root id={root!r} facet={COVERAGE_CRITERIA.get(root, '')!r}>\n"
        + prompts.coverage_root_viewpoints.get(root, "") + "\n</coverage-root>\n"
        "<current-entries>\n" + entry_lines + "\n</current-entries>\n"
        "<new-evidence>\n" + _evidence_lines(list(memories)) + "\n</new-evidence>\n"
        + prompts.coverage_audit_folding_rule + " " + prompts.coverage_audit_entry_shape_rule
    )


def _structured_request(
    transport: LlmTransport, system_prompt: str, user_prompt: str, result_type: type,
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
    transport: LlmTransport, settings: ReasoningSettings, root: str,
    entries: Sequence[CoverageEntryView], memories: Sequence[MemoryView],
    *,
    store: WorldStore | None = None,
    space_id: str | None = None,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> tuple[ExtractedCoverageAudit, list[MemoryView]]:
    """Audit the largest prefix of `memories` that fits one request.

    Returns the audit together with the evidence it actually read, since the
    caller advances this root's cursor by exactly that much. A root whose
    entries alone no longer fit the budget raises rather than dropping any of
    them: compacting such a root is its own pass, not a silent truncation.

    `entries` is stable-resorted by similarity to `memories` (this attempt's
    new evidence) first -- a priority signal only (see `dedup_priority`):
    every entry is still included in full, in every request, exactly as
    before. Missing `store`/`space_id`, or an unavailable embedding client,
    leaves the order untouched.
    """
    if store is not None and space_id is not None:
        entries = await similarity_priority_order(
            store, space_id, "coverage_entry", entries, lambda item: item.id,
            "\n".join(memory.content for memory in memories),
            embedding_client_getter=embedding_client_getter or (lambda: None),
            instruction=settings.prompts.embedding_coverage_entry_dedup_instruction,
            timeout=embedding_query_timeout_seconds,
        )
    context_budget = transport.context_budget

    def request_for(chunk: Sequence[MemoryView]) -> ChatCompletionRequest:
        return _structured_request(
            transport, settings.prompts.coverage_audit_system,
            coverage_audit_prompt(root, entries, chunk, prompts=settings.prompts),
            ExtractedCoverageAudit,
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


def build_coverage_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> DescriptorReasoner:
    """Audit one root's next evidence chunk per attempt, until none is behind."""

    audit_coverage = partial(
        _audit_coverage, model, settings,
        store=store, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )

    def load_context(space_id: str):
        return store.coverage_context(space_id)

    def call(space_id: str, context):
        root, entries, memories = context
        return (
            "audit-coverage",
            partial(audit_coverage, space_id=space_id),
            (root.root, entries, memories),
        )

    def apply(space_id: str, context, result) -> bool:
        root, entries, _ = context
        audit, audited = result
        return store.apply_coverage_audit(
            space_id, root.root, root.source_watermark, root.source_cursor_id, audit,
            list(audited), {entry.id for entry in entries},
        )

    def continue_when(context, result, applied: bool) -> bool:
        # Both outcomes reload: another chunk or root is usually still
        # behind, and a lost CAS means the fresh cursor is worth re-reading.
        return True

    return DescriptorReasoner("coverage", load_context, call, apply, continue_when)


__all__ = ["build_coverage_reasoner", "coverage_audit_prompt"]
