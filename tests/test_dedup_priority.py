"""Tests for the write-time dedup priority-ordering slice.

Two tiers:

- `similarity_priority_order` itself, unit-tested against a minimal
  duck-typed store double (only `search_vectors` matters to it) plus
  `tests/fakes_embedding.py`'s `FakeEmbeddingClient` -- covers every
  degrade path and the "reorder, never filter" contract directly.
- The three call sites (owner.py's hypothesis comparisons, coverage.py's
  active entries, learning_goals.py's open goals), each checked for (a)
  byte-identical order to before this slice when no client is configured,
  and (b) that reordering actually changes what a caller's own existing
  budget/rendering logic does with the set -- owner.py's is the only one
  of the three with real budget-driven skeleton downgrade today, so that
  is where "most similar keeps full text" is meaningfully tested; the
  other two only have order to observe (see the module docstring in
  `dedup_priority.py`).
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import (
    CoverageEntryView,
    HypothesisView,
    LearningGoalView,
    PersonProjectionResult,
    PersonView,
    ReasoningActionsResult,
)
from gossipmemo.reasoners.coverage import _audit_coverage
from gossipmemo.reasoners.dedup_priority import similarity_priority_order
from gossipmemo.reasoners.learning_goals import _root_candidates
from gossipmemo.reasoners.owner import _bounded_comparisons
from gossipmemo.reasoners.person import _reason_person
from gossipmemo.reasoners.settings import ReasoningSettings
from tests.fakes_embedding import FakeEmbeddingClient

REASONING = ReasoningSettings()
PROMPTS = REASONING.prompts


class _StubVectorStore:
    """Duck-typed `WorldStore` double: only `search_vectors` is exercised."""

    def __init__(
        self, ranking: dict[str, list[tuple[str, float]]] | None = None, raises: bool = False,
    ) -> None:
        self.ranking = ranking or {}
        self.raises = raises
        self.calls: list[tuple[str, str, int]] = []

    def search_vectors(self, space_id, owner_kind, query_vector, k, *, statuses=None):
        self.calls.append((space_id, owner_kind, k))
        if self.raises:
            raise RuntimeError("search_vectors boom")
        return self.ranking.get(owner_kind, [])


class _RaisingEmbeddingClient:
    model = "raising"
    dim = 8

    async def embed(self, texts, *, instruction=None):
        raise RuntimeError("embed boom")


def _fake_handler(calls: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    return handler


# =====================================================================
# similarity_priority_order -- unit tests
# =====================================================================


def test_no_client_returns_original_order():
    store = _StubVectorStore(ranking={"hypothesis": [("b", 0.9), ("a", 0.5)]})
    result = asyncio.run(similarity_priority_order(
        store, "s1", "hypothesis", ["a", "b", "c"], lambda item: item, "evidence text",
        embedding_client_getter=lambda: None, instruction="find dupes",
    ))
    assert result == ["a", "b", "c"]
    assert store.calls == []


def test_blank_query_text_returns_original_order():
    store = _StubVectorStore(ranking={"hypothesis": [("b", 0.9), ("a", 0.5)]})
    client = FakeEmbeddingClient()
    result = asyncio.run(similarity_priority_order(
        store, "s1", "hypothesis", ["a", "b"], lambda item: item, "   ",
        embedding_client_getter=lambda: client, instruction="find dupes",
    ))
    assert result == ["a", "b"]
    assert client.calls == []


def test_single_item_returns_unchanged_without_calling_anything():
    store = _StubVectorStore()
    client = FakeEmbeddingClient()
    result = asyncio.run(similarity_priority_order(
        store, "s1", "hypothesis", ["only"], lambda item: item, "evidence",
        embedding_client_getter=lambda: client, instruction="find dupes",
    ))
    assert result == ["only"]
    assert client.calls == []


def test_embedding_failure_falls_back_to_original_order(caplog):
    store = _StubVectorStore(ranking={"hypothesis": [("b", 0.9), ("a", 0.5)]})
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(similarity_priority_order(
            store, "s1", "hypothesis", ["a", "b"], lambda item: item, "evidence",
            embedding_client_getter=lambda: _RaisingEmbeddingClient(), instruction="find dupes",
        ))
    assert result == ["a", "b"]
    assert store.calls == []  # embed_query_vector degraded before search_vectors was ever tried


def test_search_vectors_raising_falls_back_to_original_order(caplog):
    store = _StubVectorStore(raises=True)
    client = FakeEmbeddingClient()
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(similarity_priority_order(
            store, "s1", "hypothesis", ["a", "b", "c"], lambda item: item, "evidence",
            embedding_client_getter=lambda: client, instruction="find dupes",
        ))
    assert result == ["a", "b", "c"]
    assert any("dedup_priority_search_failed" in record.message for record in caplog.records)


def test_reorders_by_rank_and_keeps_unmatched_items_in_relative_order():
    store = _StubVectorStore(ranking={"hypothesis": [("c", 0.9), ("a", 0.5)]})
    client = FakeEmbeddingClient()
    # "b" and "d" never appear in the ranking -- they must still be present,
    # trailing the ranked ones, in their original relative order.
    result = asyncio.run(similarity_priority_order(
        store, "s1", "hypothesis", ["a", "b", "c", "d"], lambda item: item, "evidence",
        embedding_client_getter=lambda: client, instruction="find dupes",
    ))
    assert result == ["c", "a", "b", "d"]


def test_uses_the_query_side_instruction_and_owner_kind():
    store = _StubVectorStore(ranking={"hypothesis": [("a", 1.0)]})
    client = FakeEmbeddingClient()
    asyncio.run(similarity_priority_order(
        store, "s1", "hypothesis", ["a", "b"], lambda item: item, "the evidence text",
        embedding_client_getter=lambda: client, instruction="a distinctive instruction",
    ))
    assert client.calls == [(("the evidence text",), "a distinctive instruction")]
    assert store.calls == [("s1", "hypothesis", 200)]


# =====================================================================
# owner.py: hypothesis dedup -- real budget-driven skeleton downgrade
# =====================================================================


def _memory(identifier: str, content: str):
    from gossipmemo.models import MemoryView

    return MemoryView(
        id=identifier, content=content, kind="fact", basis="stated", status="active",
        created_at="t", updated_at="t",
    )


def _hypothesis(identifier: str, content: str) -> HypothesisView:
    return HypothesisView(
        id=identifier, space_id="s1", owner_kind="person", owner_id="p", content=content,
        kind="fact", confidence="low", status="open", created_at="t", updated_at="t",
    )


def test_bounded_comparisons_expands_earlier_items_first_baseline() -> None:
    """Establishes the pre-slice-5 shape this test module builds on: order
    of the `hypotheses` argument alone decides who keeps full content under
    a budget too tight for all three."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    budget = ContextBudget(4300, 200, 100)
    adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client, context_budget=budget)
    person = PersonView(id="p", display_name="Bob")
    hyps = [
        _hypothesis("h1", "A" * 2000), _hypothesis("h2", "B" * 2000), _hypothesis("h3", "C" * 2000),
    ]

    _, chosen = _bounded_comparisons(
        adapter, REASONING, REASONING.prompts.person_reasoning_system, person, [], hyps,
        PersonProjectionResult, ReasoningActionsResult,
    )
    kept_full = {item.id for item in chosen if item.content}
    assert kept_full == {"h1", "h2"}  # first two in argument order, not h3


def test_owner_reasoning_priority_reorders_hypotheses_by_similarity_to_evidence() -> None:
    """With store+client wired, `owner_reasoning` reorders `hypotheses`
    before `_bounded_comparisons` sees them, so the search-ranked item
    (h3) keeps its full content and the argument-order winner (h2) is the
    one downgraded to a skeleton instead."""

    calls: list[dict] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler(calls)))
    budget = ContextBudget(4300, 200, 100)
    adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client, context_budget=budget)
    hyps = [
        _hypothesis("h1", "A" * 2000), _hypothesis("h2", "B" * 2000), _hypothesis("h3", "C" * 2000),
    ]
    store = _StubVectorStore(ranking={"hypothesis": [("h3", 0.9)]})

    asyncio.run(_reason_person(
        adapter, REASONING, PersonView(id="p", display_name="Bob"), [_memory("m1", "evidence")],
        (), hyps,
        store=store, space_id="s1", embedding_client_getter=lambda: FakeEmbeddingClient(),
    ))

    assert store.calls == [("s1", "hypothesis", 200)]
    # The final request (the actions-stage call) replays the whole first
    # message as an assistant turn, so the comparison-only hypothesis block
    # is still there to inspect for which ids kept content (up to
    # `_bounded_comparisons`'s 1200-char expansion cap) versus a skeleton
    # (empty content).
    text = json.dumps(calls[-1])
    assert "C" * 1200 in text
    assert "B" * 1200 not in text


def test_owner_reasoning_without_store_or_client_matches_pre_slice_order() -> None:
    """Regression guard: omitting `store`/`space_id`/`embedding_client_getter`
    (the pre-slice-5 call shape) produces byte-identical requests to what
    `_bounded_comparisons` alone would build."""

    calls_with: list[dict] = []
    calls_without: list[dict] = []
    budget = ContextBudget(4300, 200, 100)
    hyps = [
        _hypothesis("h1", "A" * 2000), _hypothesis("h2", "B" * 2000), _hypothesis("h3", "C" * 2000),
    ]
    person = PersonView(id="p", display_name="Bob")

    async def run(calls: list[dict], **kwargs) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler(calls)))
        adapter = OpenAICompatibleAdapter(
            "http://x", "k", "m", client=client, context_budget=budget,
        )
        await _reason_person(adapter, REASONING, person, [], (), hyps, **kwargs)

    asyncio.run(run(calls_with, store=None, space_id="s1", embedding_client_getter=None))
    asyncio.run(run(calls_without))
    assert calls_with == calls_without


# =====================================================================
# coverage.py: active-entries ordering (no existing per-item downgrade
# for entries -- see this module's docstring)
# =====================================================================


def _entry(identifier: str, content: str) -> CoverageEntryView:
    return CoverageEntryView(
        id=identifier, space_id="s1", root="M1", path="", content=content,
        created_at="t", updated_at="t",
    )


def test_coverage_entries_reorder_by_similarity_to_new_evidence() -> None:
    from gossipmemo.models import MemoryView

    calls: list[dict] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler(calls)))
    adapter = OpenAICompatibleAdapter(
        "http://x", "k", "m", client=client, context_budget=ContextBudget(20000, 200, 100),
    )
    entries = [_entry("e1", "first"), _entry("e2", "second"), _entry("e3", "third")]
    memory = MemoryView(
        id="m1", content="new fact", kind="fact", basis="stated", status="active",
        created_at="t", updated_at="t",
    )
    store = _StubVectorStore(ranking={"coverage_entry": [("e3", 0.9)]})

    asyncio.run(_audit_coverage(
        adapter, REASONING, "M1", entries, [memory],
        store=store, space_id="s1", embedding_client_getter=lambda: FakeEmbeddingClient(),
    ))

    assert store.calls == [("s1", "coverage_entry", 200)]
    prompt = calls[0]["messages"][-1]["content"]
    assert prompt.index("id='e3'") < prompt.index("id='e1'")
    assert prompt.index("id='e3'") < prompt.index("id='e2'")


def test_coverage_entries_without_store_or_client_matches_pre_slice_order() -> None:
    from gossipmemo.models import MemoryView

    entries = [_entry("e1", "first"), _entry("e2", "second"), _entry("e3", "third")]
    memory = MemoryView(
        id="m1", content="new fact", kind="fact", basis="stated", status="active",
        created_at="t", updated_at="t",
    )

    async def run(**kwargs) -> list[dict]:
        calls: list[dict] = []
        client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler(calls)))
        adapter = OpenAICompatibleAdapter(
            "http://x", "k", "m", client=client, context_budget=ContextBudget(20000, 200, 100),
        )
        await _audit_coverage(adapter, REASONING, "M1", entries, [memory], **kwargs)
        return calls

    calls_with = asyncio.run(run(store=None, space_id="s1", embedding_client_getter=None))
    calls_without = asyncio.run(run())
    assert calls_with == calls_without


# =====================================================================
# learning_goals.py: open-goal ordering (also no existing per-item
# downgrade -- open goals are always sent whole)
# =====================================================================


def _goal(identifier: str, prompt: str) -> LearningGoalView:
    return LearningGoalView(
        id=identifier, space_id="s1", prompt=prompt, rationale=prompt,
        status="open", created_at="t", updated_at="t",
    )


def test_learning_goals_reorder_by_similarity_to_root_entries() -> None:
    calls: list[dict] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler(calls)))
    adapter = OpenAICompatibleAdapter(
        "http://x", "k", "m", client=client, context_budget=ContextBudget(20000, 200, 100),
    )
    entries = [_entry("e1", "life chapter overview")]
    goals = [_goal("g1", "first goal"), _goal("g2", "second goal"), _goal("g3", "third goal")]
    store = _StubVectorStore(ranking={"learning_goal": [("g3", 0.9)]})

    asyncio.run(_root_candidates(
        adapter, REASONING, "M1", entries, goals,
        store=store, space_id="s1", embedding_client_getter=lambda: FakeEmbeddingClient(),
    ))

    assert store.calls == [("s1", "learning_goal", 200)]
    prompt = calls[0]["messages"][-1]["content"]
    assert prompt.index("id='g3'") < prompt.index("id='g1'")
    assert prompt.index("id='g3'") < prompt.index("id='g2'")


def test_learning_goals_without_store_or_client_matches_pre_slice_order() -> None:
    entries = [_entry("e1", "life chapter overview")]
    goals = [_goal("g1", "first goal"), _goal("g2", "second goal"), _goal("g3", "third goal")]

    async def run(**kwargs) -> list[dict]:
        calls: list[dict] = []
        client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler(calls)))
        adapter = OpenAICompatibleAdapter(
            "http://x", "k", "m", client=client, context_budget=ContextBudget(20000, 200, 100),
        )
        await _root_candidates(adapter, REASONING, "M1", entries, goals, **kwargs)
        return calls

    calls_with = asyncio.run(run(store=None, space_id="s1", embedding_client_getter=None))
    calls_without = asyncio.run(run())
    assert calls_with == calls_without


# =====================================================================
# instruction wiring: each site asks for its own dedup instruction
# =====================================================================


def test_each_call_site_uses_its_own_dedup_instruction() -> None:
    seen: dict[str, str] = {}

    class _RecordingClient(FakeEmbeddingClient):
        async def embed(self, texts, *, instruction=None):
            seen[texts[0]] = instruction
            return await super().embed(texts, instruction=instruction)

    async def run() -> None:
        store = _StubVectorStore()
        client = _RecordingClient()
        adapter = OpenAICompatibleAdapter(
            "http://x", "k", "m",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_fake_handler([]))),
            context_budget=ContextBudget(20000, 200, 100),
        )
        from gossipmemo.models import MemoryView
        await _reason_person(
            adapter, REASONING, PersonView(id="p", display_name="Bob"), [_memory("m1", "content")],
            (), [_hypothesis("h1", "irrelevant"), _hypothesis("h2", "also irrelevant")],
            store=store, space_id="s1", embedding_client_getter=lambda: client,
        )
        await _audit_coverage(
            adapter, REASONING, "M1", [_entry("e1", "x"), _entry("e2", "y")],
            [MemoryView(id="m1", content="new fact", kind="fact", basis="stated",
                        status="active", created_at="t", updated_at="t")],
            store=store, space_id="s1", embedding_client_getter=lambda: client,
        )
        await _root_candidates(
            adapter, REASONING, "M1", [_entry("e1", "x")],
            [_goal("g1", "a"), _goal("g2", "b")],
            store=store, space_id="s1", embedding_client_getter=lambda: client,
        )

    asyncio.run(run())
    assert seen == {
        "content": PROMPTS.embedding_hypothesis_dedup_instruction,
        "new fact": PROMPTS.embedding_coverage_entry_dedup_instruction,
        ": x": PROMPTS.embedding_learning_goal_dedup_instruction,
    }
