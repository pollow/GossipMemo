"""Tests for the narrow `LlmTransport` seam alongside the wide `LlmModel`.

Focused on the new pieces only: `OpenAICompatibleAdapter.complete` (gate
acquisition + hard budget check) and the module-level `structured()` helper
(malformed-output retry). Existing `LlmModel` reasoner behavior is covered
elsewhere and must stay unchanged.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import (
    ChatCompletionRequest,
    ChatMessage,
    OpenAICompatibleAdapter,
    structured,
)
from gossipmemo.models import ExtractionResult


def test_complete_acquires_gate_around_the_request() -> None:
    """`complete()` holds the gate for the duration of the HTTP call."""

    observed_in_flight: list[bool] = []
    adapter_holder: dict[str, OpenAICompatibleAdapter] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed_in_flight.append(adapter_holder["adapter"].gate.in_flight)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            adapter_holder["adapter"] = adapter
            request = ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="hi")],
            )
            assert adapter.gate.in_flight is False
            content = await adapter.complete(request)
            assert adapter.gate.in_flight is False
            return content

    content = asyncio.run(run())
    assert content == "ok"
    # The gate was held (locked) while the mocked handler ran.
    assert observed_in_flight == [True]


def test_complete_enforces_hard_context_budget_before_sending() -> None:
    """`complete()` rejects an oversized request without any HTTP call."""

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    budget = ContextBudget(200, 50, 20)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, context_budget=budget,
            )
            request = ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x" * 5000)],
            )
            with pytest.raises(ValueError, match="LLM context exceeds input budget"):
                await adapter.complete(request)

    asyncio.run(run())
    assert calls == []


def test_structured_retries_malformed_output_then_returns_parsed_model() -> None:
    """`structured()` retries `LLMOutputError` and returns the parsed result."""

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            content = "not json"
        else:
            content = json.dumps({"memories": [], "people": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def run() -> tuple[str, ExtractionResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                max_retries=2, retry_base_seconds=0.001, retry_max_seconds=0.002,
            )
            messages = [
                ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="usr"),
            ]
            return await structured(
                adapter, messages, ExtractionResult, tier=3, label="test",
            )

    content, result = asyncio.run(run())
    assert len(calls) == 2
    assert isinstance(result, ExtractionResult)
    assert result.memories == [] and result.people == []
    assert content == json.dumps({"memories": [], "people": []})


def test_structured_raises_after_exhausting_retries() -> None:
    """`structured()` propagates `LLMOutputError` once retries are exhausted."""

    from gossipmemo.llm import LLMOutputError

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "still not json"}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                max_retries=1, retry_base_seconds=0.001, retry_max_seconds=0.002,
            )
            messages = [ChatMessage(role="user", content="usr")]
            with pytest.raises(LLMOutputError):
                await structured(
                    adapter, messages, ExtractionResult, tier=3, label="test",
                )

    asyncio.run(run())
    assert len(calls) == 2  # initial attempt + 1 retry
