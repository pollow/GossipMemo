from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import (
    HypothesisView,
    ManualMemoryRequest,
    MemoryView,
    PersonReasoningResult,
    PersonView,
)
from gossipmemo.reasoners import ReasoningSettings
from gossipmemo.reasoners.person import _reason_person
from gossipmemo.store import SqliteWorldStore
from gossipmemo.transport import ChatCompletionRequest

REASONING = ReasoningSettings()


def _memory(
    identifier: str, content: str, *, basis: str = "stated", status: str = "active"
) -> MemoryView:
    return MemoryView(
        id=identifier,
        content=content,
        kind="fact",
        basis=basis,
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _prefix(payload: dict) -> str:
    """The owner-reasoning prefix, which is always the second message."""
    return str(payload["messages"][1]["content"])


def _fold_handler(calls: list[dict], cards: list[str]):
    """Answer a projection call with the next card, an actions call with one
    hypothesis upsert, so a fold over several batches is countable."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        combined = str(payload["messages"][-1]["content"])
        if "Return only the requested projection" in combined:
            body = {"profile_card": {"summary": cards[
                sum("Return only the requested projection" in
                    str(call["messages"][-1]["content"]) for call in calls) - 1]}}
        else:
            body = {"hypothesis_actions": {"upserts": [{
                "content": "tentative",
                "evidence": [{"memory_id": "m0", "role": "support"}],
            }], "transitions": []}}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    return handler


def test_owner_fold_over_several_batches_carries_the_card_into_the_next_one() -> None:
    """A rebuild has no watermark, so the delta is the whole history: it is
    folded batch by batch, each batch reading the card the last one wrote."""

    calls: list[dict] = []
    cards = [f"card{index}" for index in range(20)]
    budget = ContextBudget(6000, 400, 200)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_fold_handler(calls, cards))) \
                as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget)
            result = await _reason_person(
                adapter,
                REASONING,
                PersonView(id="p", display_name="Bob"),
                [_memory(f"m{index}", "往事" * 400) for index in range(12)],
            )
            projections = sum(
                "Return only the requested projection" in str(call["messages"][-1]["content"])
                for call in calls)
            assert projections >= 3
            assert result.profile_card == {"summary": cards[projections - 1]}
            # One epistemic review per batch, and every batch's actions kept.
            assert len(calls) == projections * 2
            assert result.hypothesis_actions is not None
            assert len(result.hypothesis_actions.upserts) == projections

    asyncio.run(run())
    projection_calls = [
        call for call in calls
        if "Return only the requested projection" in str(call["messages"][-1]["content"])]
    # The first batch folds into the stored (empty) card; every later batch
    # folds into the card its predecessor produced.
    assert "card0" not in _prefix(projection_calls[0])
    for index, call in enumerate(projection_calls[1:]):
        assert f'"{cards[index]}"' in _prefix(call)
    # Each memory reaches exactly one batch, and nothing was summarized away.
    seen = [identifier for call in projection_calls
            for identifier in re.findall(r"- id='(m\d+)'", _prefix(call))]
    assert sorted(seen) == sorted(f"m{index}" for index in range(12))
    for payload in calls:
        request = ChatCompletionRequest.model_validate(payload)
        assert budget.estimate_request(request) <= budget.usable_input_tokens


def test_owner_fold_of_a_steady_state_delta_is_a_single_pair() -> None:
    """The normal case: a handful of new memories folded into a real card."""

    calls: list[dict] = []

    async def run() -> None:
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(_fold_handler(calls, ["next"]))) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            result = await _reason_person(
                adapter,
                REASONING,
                PersonView(id="p", display_name="Bob",
                           profile_card={"summary": "prefers tea"},
                           profile_source_updated_at="2026-01-01T00:00:00+00:00"),
                [_memory("m1", "Bob switched to coffee.")],
            )
            assert result.profile_card == {"summary": "next"}

    asyncio.run(run())
    assert len(calls) == 2
    prefix = _prefix(calls[0])
    assert "prefers tea" in prefix
    assert "- id='m1'" in prefix


def test_owner_fold_renders_invalidated_memories_in_their_own_section() -> None:
    """Retracted and superseded rows travel in the delta as a negative
    instruction, never as an evidence line carrying a flag."""

    calls: list[dict] = []

    async def run() -> None:
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(_fold_handler(calls, ["next"]))) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            await _reason_person(
                adapter,
                REASONING,
                PersonView(id="p", display_name="Bob", profile_card={"summary": "drinks tea"}),
                [
                    _memory("m1", "Bob drinks tea.", status="retracted"),
                    _memory("m2", "Bob drinks coffee."),
                ],
            )

    asyncio.run(run())
    prefix = _prefix(calls[0])
    evidence = prefix.split("<evidence-memories>")[1].split("</evidence-memories>")[0]
    invalidated = prefix.split("<invalidated-memories")[1].split("</invalidated-memories>")[0]
    assert "m2" in evidence and "m1" not in evidence
    assert "m1" in invalidated and "m2" not in invalidated
    assert "status='retracted'" in invalidated
    # The evidence line is id/kind/basis/text and nothing else.
    assert "derivation_sources" not in prefix


def test_person_delta_read_carries_a_retraction_and_marks_the_card_stale(tmp_path) -> None:
    store = SqliteWorldStore(tmp_path / "fold.db")
    store.initialize()
    kept = store.add_manual_memory(
        "s", ManualMemoryRequest(content="Bob drinks tea.", people=["Bob"]))
    with store._connect() as connection:
        person_id = connection.execute(
            "SELECT id FROM people WHERE space_id = 's'").fetchone()["id"]
    person, memories, watermark = store.person_context("s", person_id, delta_only=True)
    assert [memory.id for memory in memories] == [kept]
    assert store.apply_person_reasoning(
        "s", person_id, watermark, PersonReasoningResult(profile_card={"summary": "tea"})) is True
    # Folded up to the watermark: nothing new, nothing to do.
    person, memories, _ = store.person_context("s", person_id, delta_only=True)
    assert memories == [] and person.stale is False

    store.retract_memory("s", kept)
    person, memories, _ = store.person_context("s", person_id, delta_only=True)
    assert person.stale is True
    assert [(memory.id, memory.status) for memory in memories] == [(kept, "retracted")]
    # The dossier read is unchanged: active memories only, newest first.
    assert store.person_context("s", person_id)[1] == []


def test_owner_stage_two_checks_actual_first_completion_before_second_http() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({
                "profile_card": {"summary": "大" * 10000},
            })}}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                context_budget=ContextBudget(5000, 300, 200),
            )
            with pytest.raises(ValueError, match="LLM context exceeds input budget"):
                await _reason_person(adapter, REASONING, PersonView(id="p", display_name="Bob"), [])

    asyncio.run(run())
    assert calls == 1


def test_owner_reasoning_retries_malformed_structured_output() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            content = '{"profile_card" "malformed"}'
        elif calls == 2:
            content = '{"profile_card":{"summary":"ok"}}'
        else:
            content = '{"hypothesis_actions":{"upserts":[],"transitions":[]}}'
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                max_retries=1, retry_base_seconds=0.001,
                retry_max_seconds=0.001,
            )
            result = await _reason_person(
                adapter,
                REASONING,
                PersonView(id="p", display_name="Bob"), [],
            )
            assert result.profile_card == {"summary": "ok"}

    asyncio.run(run())
    assert calls == 3


def test_owner_comparison_state_is_bounded_and_remains_comparison_only() -> None:
    calls: list[dict] = []
    budget = ContextBudget(6000, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        combined = str(payload["messages"])
        body = (
            {"profile_card": {}}
            if "Return only the requested projection" in combined
            else {"hypothesis_actions": {"upserts": [], "transitions": []}}
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
        )

    inferred = [_memory(f"i{index}", "隐私" * 5000, basis="inferred") for index in range(100)]
    hypotheses = [
        HypothesisView(
            id=f"h{index}", space_id="s", owner_kind="person", owner_id="p",
            content="可能" * 5000, kind="impression", confidence="low", status="open",
            created_at="1", updated_at="1",
        )
        for index in range(100)
    ]

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget,
            )
            await _reason_person(
                adapter,
                REASONING,
                PersonView(id="p", display_name="Bob"), [], inferred, hypotheses,
            )

    asyncio.run(run())
    assert len(calls) == 2
    assert "comparison-only" in str(calls[0])
    assert "隐私" * 100 not in str(calls[0])
    for payload in calls:
        assert budget.estimate_request(ChatCompletionRequest.model_validate(
            payload)) <= budget.usable_input_tokens


def test_owner_store_snapshot_and_authority_include_more_than_one_hundred_memories(
    tmp_path,
) -> None:
    store = SqliteWorldStore(tmp_path / "owner.db")
    store.initialize()
    for index in range(105):
        store.add_manual_memory(
            "s",
            ManualMemoryRequest(content=f"Bob event {index}", people=["Bob"]),
        )
    with store._connect() as connection:
        person_id = connection.execute(
            "SELECT id FROM people WHERE space_id = 's'"
        ).fetchone()["id"]
        authority = store._person_reasoning_source_ids(connection, "s", person_id)
    context = store.person_context("s", person_id)
    assert context is not None
    _, memories, _ = context
    assert len(memories) == 105
    assert authority == {memory.id for memory in memories}
