from __future__ import annotations

import asyncio
import json

import httpx

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import ChatCompletionRequest, OpenAICompatibleAdapter
from gossipmemo.models import (
    ContinuityView, CoverageMapView, HypothesisView, LearningGoalView,
    MemoryView, ModelMessage,
)


def _memory(identifier: str, content: str) -> MemoryView:
    return MemoryView(id=identifier, content=content, kind="fact", basis="stated", status="active", created_at="1")


def _hypothesis(identifier: str, content: str) -> HypothesisView:
    return HypothesisView(id=identifier, space_id="s", owner_kind="user", content=content, kind="fact", confidence="low", status="open", created_at="1", updated_at="1")


def _message(identifier: str, content: str) -> ModelMessage:
    return ModelMessage(id=identifier, space_id="s", author="user", content=content, occurred_at="1", source_provider="test")


def test_small_paths_send_original_single_requests() -> None:
    calls: list[dict] = []
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content); calls.append(payload)
        prompt = str(payload["messages"])
        body = {"criteria": []} if "<new-evidence>" in prompt else {"text": "ok", "related_person_ids": [], "through_message_id": "m"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            coverage = CoverageMapView(space_id="s")
            await adapter.audit_coverage(coverage, [_memory("e", "raw evidence")], [_hypothesis("h", "raw hypothesis")])
            await adapter.reason_continuity(ContinuityView(text="prior"), [_message("m", "raw message")])
    asyncio.run(run())
    assert len(calls) == 2
    assert "raw hypothesis" in str(calls[0]) and "raw evidence" in str(calls[0])
    assert "raw message" in str(calls[1]) and "prior" in str(calls[1])


def test_coverage_cjk_chunks_filter_fake_evidence_ids() -> None:
    calls: list[dict] = []; budget = ContextBudget(6500, 400, 200)
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content); calls.append(payload)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"criteria": [{"criterion_id": "M1", "level": "grounded", "evidence_memory_ids": ["forged", "m0"]}], "boundary_upserts": [{"kind": "blind_spot", "summary": "x", "evidence_memory_ids": ["forged", "m0"]}]})}}]})
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await OpenAICompatibleAdapter("http://x", "k", "m", client=client, context_budget=budget).audit_coverage(CoverageMapView(space_id="s"), [_memory(f"m{i}", "证据" * 2500) for i in range(3)])
            assert all("forged" not in item.evidence_memory_ids for item in result.criteria + result.boundary_upserts)
    asyncio.run(run())
    assert len(calls) > 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item)) <= budget.usable_input_tokens for item in calls)


def test_goal_overflow_uses_multiple_candidate_calls_and_one_final() -> None:
    calls: list[dict] = []; budget = ContextBudget(7000, 400, 200)
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content); calls.append(payload); prompt = str(payload["messages"])
        body = {"candidates": []} if "Propose optional candidate invitations only" in prompt else {"upserts": [], "transitions": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await OpenAICompatibleAdapter("http://x", "k", "m", client=client, context_budget=budget).plan_learning_goals(CoverageMapView(space_id="s"), [_hypothesis(f"h{i}", "假设" * 2500) for i in range(4)], [], [])
    asyncio.run(run())
    candidates = [item for item in calls if "Propose optional candidate invitations only" in str(item)]
    finals = [item for item in calls if "<candidates>" in str(item)]
    assert len(candidates) > 1 and len(finals) == 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item)) <= budget.usable_input_tokens for item in calls)


def test_continuity_oversized_cjk_streams_and_preserves_last_id() -> None:
    calls: list[dict] = []; budget = ContextBudget(6500, 400, 200)
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content); calls.append(payload)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"text": "ok", "related_person_ids": ["forged"], "through_message_id": "wrong"})}}]})
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await OpenAICompatibleAdapter("http://x", "k", "m", client=client, context_budget=budget).reason_continuity(ContinuityView(), [_message("m0", "消息" * 9000), _message("m1", "消息" * 9000)])
            assert result.through_message_id == "m1"
    asyncio.run(run())
    assert len(calls) > 1
    assert all(budget.estimate_request(ChatCompletionRequest.model_validate(item)) <= budget.usable_input_tokens for item in calls)
