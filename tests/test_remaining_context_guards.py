"""Budget guards on the reasoners that split oversized context.

These drive each reasoner's own entry point rather than the world, so an
adapter can be given a deliberately tiny `ContextBudget` and every issued
request inspected. Coverage, goal planning, continuity, and the owner
family all live in `reasoners/` and are driven directly here; query
synthesis has no oversized-input guard of its own (it is one bounded,
unpaginated call), so it stays untested by this file.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import (
    ContinuityView,
    CoverageEntryView,
    MemoryView,
    ModelMessage,
)
from gossipmemo.reasoners.continuity import _reason_continuity
from gossipmemo.reasoners.coverage import _audit_coverage
from gossipmemo.reasoners.learning_goals import _plan_learning_goals
from gossipmemo.transport import ChatCompletionRequest


def _memory(identifier: str, content: str) -> MemoryView:
    return MemoryView(id=identifier, content=content, kind="fact", basis="stated",
                      status="active", created_at="1")


def _entry(identifier: str, content: str, *, root: str = "M1", path: str = "") -> CoverageEntryView:
    return CoverageEntryView(id=identifier, space_id="s", root=root, path=path, content=content,
                             created_at="1", updated_at="1")


def _message(identifier: str, content: str) -> ModelMessage:
    return ModelMessage(id=identifier, space_id="s", author="user", content=content,
                        occurred_at="1", source_provider="test")


def test_small_paths_send_original_single_requests() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        prompt = str(payload["messages"])
        body = {"additions": []} if "<new-evidence>" in prompt else {
            "text": "ok", "related_person_ids": [], "through_message_id": "m"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            await _audit_coverage(adapter, "M1", [_entry("c", "prior understanding")],
                                  [_memory("e", "raw evidence")])
            await _reason_continuity(
                adapter, ContinuityView(text="prior"), [_message("m", "raw message")])
    asyncio.run(run())
    assert len(calls) == 2
    assert "prior understanding" in str(calls[0]) and "raw evidence" in str(calls[0])
    assert "raw message" in str(calls[1]) and "prior" in str(calls[1])


def test_coverage_cjk_backlog_audits_one_budget_sized_chunk() -> None:
    calls: list[dict] = []
    budget = ContextBudget(6500, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        content = json.dumps({"additions": [{"path": "", "content": "总结"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget)
            memories = [_memory(f"m{i}", "证据" * 2500) for i in range(3)]
            _, audited = await _audit_coverage(adapter, "M1", [], memories)
            # One request per attempt; the caller only commits the evidence
            # this request actually read, so the rest stays in the backlog.
            assert 0 < len(audited) < len(memories)
    asyncio.run(run())
    assert len(calls) == 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item))
               <= budget.usable_input_tokens for item in calls)


def test_goal_planning_fans_out_per_root_and_reconciles_once() -> None:
    """Every root gets its own request, and only one pass may mutate goals.

    Fan-out is the normal path here, not an overflow path: M2 has one small
    entry and still gets a request of its own. Oversized entries only add
    chunks *within* a root, and the root's overview entry rides along in
    each of them.
    """
    calls: list[dict] = []
    budget = ContextBudget(7000, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        prompt = str(payload["messages"])
        body = {"candidates": [{"prompt": "问", "rationale": "因"}]} if (
            "Propose optional candidate directions only" in prompt) else {
            "upserts": [], "transitions": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget)
            entries = [
                _entry("m1-overview", "已知章节：求学、迁移。"),
                *(_entry(f"m1-{i}", "证据" * 2500, path=f"阶段{i}") for i in range(3)),
                _entry("m2-overview", "日常生活的轮廓已知。", root="M2"),
            ]
            await _plan_learning_goals(adapter, entries, [], [])
    asyncio.run(run())
    prompts = [" ".join(message["content"] for message in item["messages"]) for item in calls]
    first_root = [item for item in prompts if "id='M1'" in item]
    second_root = [item for item in prompts if "id='M2'" in item]
    finals = [item for item in prompts if "<candidates>" in item]
    assert len(first_root) > 1 and len(second_root) == 1 and len(finals) == 1
    assert all("已知章节：求学、迁移。" in item for item in first_root)
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item))
               <= budget.usable_input_tokens for item in calls)


def test_continuity_oversized_cjk_streams_and_preserves_last_id() -> None:
    calls: list[dict] = []
    budget = ContextBudget(6500, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        content = json.dumps(
            {"text": "ok", "related_person_ids": ["forged"], "through_message_id": "wrong"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget)
            result = await _reason_continuity(
                adapter, ContinuityView(),
                [_message("m0", "消息" * 9000), _message("m1", "消息" * 9000)])
            assert result.through_message_id == "m1"
    asyncio.run(run())
    assert len(calls) > 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item))
               <= budget.usable_input_tokens for item in calls)
