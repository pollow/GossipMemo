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
from gossipmemo.transport import ChatCompletionRequest
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import (
    ContinuityView, CoverageEntryView, HypothesisView, LearningGoalView,
    MemoryView, ModelMessage,
)
from gossipmemo.reasoners.continuity import _reason_continuity
from gossipmemo.reasoners.coverage import _audit_coverage
from gossipmemo.reasoners.learning_goals import _plan_learning_goals


def _memory(identifier: str, content: str) -> MemoryView:
    return MemoryView(id=identifier, content=content, kind="fact", basis="stated", status="active", created_at="1")


def _entry(identifier: str, content: str) -> CoverageEntryView:
    return CoverageEntryView(id=identifier, space_id="s", root="M1", content=content, created_at="1", updated_at="1")


def _hypothesis(identifier: str, content: str) -> HypothesisView:
    return HypothesisView(id=identifier, space_id="s", owner_kind="user", content=content, kind="fact", confidence="low", status="open", created_at="1", updated_at="1")


def _message(identifier: str, content: str) -> ModelMessage:
    return ModelMessage(id=identifier, space_id="s", author="user", content=content, occurred_at="1", source_provider="test")


def test_small_paths_send_original_single_requests() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        prompt = str(payload["messages"])
        body = {"additions": []} if "<new-evidence>" in prompt else {"text": "ok",
                                                                     "related_person_ids": [], "through_message_id": "m"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            await _audit_coverage(adapter, "M1", [_entry("c", "prior understanding")],
                                  [_memory("e", "raw evidence")])
            await _reason_continuity(adapter, ContinuityView(text="prior"), [_message("m", "raw message")])
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
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"additions": [{"path": "", "content": "总结"}]})}}]})

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


def test_goal_overflow_uses_multiple_candidate_calls_and_one_final() -> None:
    calls: list[dict] = []
    budget = ContextBudget(7000, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        prompt = str(payload["messages"])
        body = {"candidates": []} if "Propose optional candidate invitations only" in prompt else {
            "upserts": [], "transitions": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget)
            await _plan_learning_goals(adapter, [], [_hypothesis(f"h{i}", "假设" * 2500) for i in range(4)], [], [])
    asyncio.run(run())
    candidates = [
        item for item in calls if "Propose optional candidate invitations only" in str(item)]
    finals = [item for item in calls if "<candidates>" in str(item)]
    assert len(candidates) > 1 and len(finals) == 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item))
               <= budget.usable_input_tokens for item in calls)


def test_continuity_oversized_cjk_streams_and_preserves_last_id() -> None:
    calls: list[dict] = []
    budget = ContextBudget(6500, 400, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"text": "ok", "related_person_ids": ["forged"], "through_message_id": "wrong"})}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget)
            result = await _reason_continuity(adapter, ContinuityView(), [_message("m0", "消息" * 9000), _message("m1", "消息" * 9000)])
            assert result.through_message_id == "m1"
    asyncio.run(run())
    assert len(calls) > 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item))
               <= budget.usable_input_tokens for item in calls)
