from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import ChatCompletionRequest, OpenAICompatibleAdapter
from gossipmemo.models import (
    HypothesisView,
    ManualMemoryRequest,
    MemoryView,
    PersonView,
)
from gossipmemo.store import SqliteWorldStore


def _memory(identifier: str, content: str, *, basis: str = "stated") -> MemoryView:
    return MemoryView(
        id=identifier,
        content=content,
        kind="fact",
        basis=basis,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_owner_chunking_filters_fake_ids_and_keeps_final_two_calls() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        text = "\n".join(str(m.get("content", "")) for m in payload["messages"])
        if "Compress evidence only" in text:
            body = {"items": [{"summary": "tea", "source_memory_ids": ["m1"]}, {"summary": "fake", "source_memory_ids": ["forged"]}]}
        elif len([c for c in calls if "Compress evidence only" not in " ".join(str(m) for m in c["messages"])]) == 1:
            body = {"profile_card": {"summary": "tea"}}
        else:
            body = {"hypothesis_actions": {"upserts": [], "transitions": []}}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client, context_budget=ContextBudget(5000, 80, 20))
            result = await adapter.reason_person(
                PersonView(id="p", display_name="Bob"),
                [_memory("m1", "x" * 10000)],
            )
            assert result.profile_card == {"summary": "tea"}

    asyncio.run(run())
    assert sum("Compress evidence only" in str(c["messages"]) for c in calls) >= 1
    assert sum("inferred_memory_actions" in str(c["messages"][-1]) for c in calls) == 1
    assert "forged" not in str(calls[-2:])


def test_owner_chunking_recursively_digests_large_cjk_with_bounded_requests() -> None:
    calls: list[dict] = []
    budget = ContextBudget(5000, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        combined = "\n".join(str(message.get("content", "")) for message in payload["messages"])
        if "Compress evidence only" in combined:
            ids = sorted(set(re.findall(r'"(m\d+)"', combined)))
            assert ids
            # The first digest round remains deliberately verbose, forcing a
            # second typed reduction while retaining original source IDs.
            summary = "摘要" if '"summary"' in combined else "中" * 500
            body = {"items": [{"summary": summary, "source_memory_ids": ids}]}
        elif "Return only the requested projection" in combined:
            body = {"profile_card": {"summary": "ok"}}
        else:
            body = {"hypothesis_actions": {"upserts": [], "transitions": []}}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget,
            )
            result = await adapter.reason_person(
                PersonView(id="p", display_name="Bob"),
                [_memory(f"m{index}", "往事" * 1200) for index in range(12)],
            )
            assert result.profile_card == {"summary": "ok"}

    asyncio.run(run())
    digest_calls = [call for call in calls if "Compress evidence only" in str(call)]
    assert len(digest_calls) > 2
    assert sum("Return only the requested projection" in str(call) for call in calls) == 2
    assert sum("inferred_memory_actions" in str(call) for call in calls) == 1
    for payload in calls:
        request = ChatCompletionRequest.model_validate(payload)
        assert budget.estimate_request(request) <= budget.usable_input_tokens


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
                await adapter.reason_person(PersonView(id="p", display_name="Bob"), [])

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
            result = await adapter.reason_person(
                PersonView(id="p", display_name="Bob"), [],
            )
            assert result.profile_card == {"summary": "ok"}

    asyncio.run(run())
    assert calls == 3


def test_owner_digest_retries_semantically_incomplete_source_ids() -> None:
    calls: list[dict] = []
    digest_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal digest_attempts
        payload = json.loads(request.content)
        calls.append(payload)
        combined = str(payload["messages"])
        if "Compress evidence only" in combined:
            digest_attempts += 1
            ids = sorted(set(re.findall(r'"(m\d+)"', combined)))
            returned = ids[:-1] if digest_attempts == 1 else ids
            body = {"items": [{
                "summary": "bounded evidence",
                "source_memory_ids": returned,
            }]}
        elif "Return only the requested projection" in combined:
            body = {"profile_card": {"summary": "ok"}}
        else:
            body = {"hypothesis_actions": {"upserts": [], "transitions": []}}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                context_budget=ContextBudget(5000, 400, 200),
                max_retries=1, retry_base_seconds=0.001,
                retry_max_seconds=0.001,
            )
            result = await adapter.reason_person(
                PersonView(id="p", display_name="Bob"),
                [_memory(f"m{index}", "证据" * 700) for index in range(6)],
            )
            assert result.profile_card == {"summary": "ok"}

    asyncio.run(run())
    digest_calls = [call for call in calls if "Compress evidence only" in str(call)]
    assert len(digest_calls) >= 2
    assert digest_calls[0]["messages"] == digest_calls[1]["messages"]


def test_recursive_digest_inherits_validated_provenance_server_side() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        combined = str(payload["messages"])
        if "Compress evidence only" in combined:
            ids = sorted(set(re.findall(r'"(m\d+)"', combined)))
            user_prompt = payload["messages"][-1]["content"]
            if '"summary"' in user_prompt:
                # A reduce response need not mechanically repeat every
                # already-validated original ID; the server inherits them.
                body = {"items": [{
                    "summary": "reduced",
                    "source_memory_ids": ["forged"],
                }]}
            else:
                groups = [ids[index:index + 2] for index in range(0, len(ids), 2)]
                body = {"items": [{
                    "summary": "中" * 600,
                    "source_memory_ids": group,
                } for group in groups]}
        elif "Return only the requested projection" in combined:
            assert "m0" in combined and "m39" in combined
            body = {"profile_card": {"summary": "ok"}}
        else:
            body = {"hypothesis_actions": {"upserts": [], "transitions": []}}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                context_budget=ContextBudget(6000, 400, 200),
            )
            result = await adapter.reason_person(
                PersonView(id="p", display_name="Bob"),
                [_memory(f"m{index}", "证据" * 300) for index in range(40)],
            )
            assert result.profile_card == {"summary": "ok"}

    asyncio.run(run())
    digest_calls = [call for call in calls if "Compress evidence only" in str(call)]
    assert any('"summary"' in str(call) for call in digest_calls)


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
            await adapter.reason_person(
                PersonView(id="p", display_name="Bob"), [], inferred, hypotheses,
            )

    asyncio.run(run())
    assert len(calls) == 2
    assert "comparison-only" in str(calls[0])
    assert "隐私" * 100 not in str(calls[0])
    for payload in calls:
        assert budget.estimate_request(ChatCompletionRequest.model_validate(payload)) <= budget.usable_input_tokens


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
