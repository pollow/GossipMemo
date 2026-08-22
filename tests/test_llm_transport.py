"""Tests for the narrow `LlmTransport` seam.

Focused on the new pieces only: `OpenAICompatibleAdapter.complete` (gate
acquisition + hard budget check) and the module-level `structured()` helper
(malformed-output retry). Per-reasoner behavior is covered elsewhere and
must stay unchanged.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import ExtractionResult
from gossipmemo.priority import TIER_BACKGROUND, llm_call_tier
from gossipmemo.transport import (
    ChatCompletionRequest,
    ChatMessage,
    LlmTransport,
    ProviderGate,
    RetryPolicy,
    structured,
)


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


class FakeTransport:
    """A transport that implements `LlmTransport` and nothing more.

    The point of this double is what it *lacks*: no `model`, no
    `max_retries`, no `_retry_delay`. If `structured()` ever reaches past
    the protocol into adapter internals again, this test fails with
    `AttributeError` instead of quietly working because the only
    implementation happened to be the real adapter.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.sent: list[ChatCompletionRequest] = []
        self._gate = ProviderGate()
        self._budget = ContextBudget()

    @property
    def configured(self) -> bool:
        return True

    @property
    def gate(self) -> ProviderGate:
        return self._gate

    @property
    def context_budget(self) -> ContextBudget:
        return self._budget

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(attempts=2, base_seconds=0.001, max_seconds=0.002)

    def prepare(
        self, messages, *, structured: bool
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake",
            messages=list(messages),
            response_format={"type": "json_object"} if structured else None,
        )

    async def complete(self, request: ChatCompletionRequest) -> str:
        self.sent.append(request)
        return self._replies.pop(0)

    async def aclose(self) -> None:
        return None


def test_structured_needs_nothing_beyond_the_protocol() -> None:
    """`structured()` drives a transport that has only the protocol members."""

    transport = FakeTransport(["not json", json.dumps({"memories": [], "people": []})])
    assert isinstance(transport, LlmTransport)

    content, result = asyncio.run(
        structured(
            transport,
            [ChatMessage(role="user", content="usr")],
            ExtractionResult,
            tier=3,
            label="test",
        )
    )

    assert len(transport.sent) == 2
    # `prepare` decided the JSON-mode request, and both attempts reused it.
    assert transport.sent[0].response_format == {"type": "json_object"}
    assert transport.sent[0] == transport.sent[1]
    assert isinstance(result, ExtractionResult)
    assert content == json.dumps({"memories": [], "people": []})


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

    from gossipmemo.transport import LLMOutputError

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


def _trace_files(trace_dir) -> list:
    """All trace JSON files under `trace_dir`, sorted by path (= chronological)."""

    return sorted(trace_dir.glob("*/*.json"))


def test_trace_records_the_verbatim_request_and_completion(tmp_path) -> None:
    """The audit seam: what went over the wire, plus the reasoner label."""

    import datetime as dt
    import re

    trace_path = tmp_path / "trace"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, trace_path=trace_path)
            request = ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="system", content="be brief"),
                          ChatMessage(role="user", content="audit M1")],
                response_format={"type": "json_object"},
            )
            with llm_call_tier(TIER_BACKGROUND, "audit-coverage"):
                await adapter.complete(request)
                await adapter.complete(request)

    asyncio.run(run())

    files = _trace_files(trace_path)
    assert len(files) == 2

    today = dt.datetime.now().astimezone().strftime("%Y-%m-%d")
    for path in files:
        assert path.parent.name == today
        assert re.fullmatch(
            r"\d{9}-audit-coverage-\d{4}(-\d+)?\.json", path.name
        ), path.name

    records = [json.loads(path.read_text()) for path in files]
    assert [item["sequence"] for item in records] == [1, 2]
    assert records[0]["label"] == "audit-coverage"
    assert records[0]["completion"] == "{}"
    assert records[0]["request"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "audit M1"},
    ]


def test_trace_files_do_not_collide_within_the_same_millisecond(tmp_path) -> None:
    """Two calls with the same label in the same millisecond both survive."""

    trace_path = tmp_path / "trace"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, trace_path=trace_path)
            request = ChatCompletionRequest(
                model="m", messages=[ChatMessage(role="user", content="hi")])
            with llm_call_tier(TIER_BACKGROUND, "same-label"):
                await asyncio.gather(*(adapter.complete(request) for _ in range(5)))

    asyncio.run(run())

    files = _trace_files(trace_path)
    assert len(files) == 5
    records = [json.loads(path.read_text()) for path in files]
    assert sorted(item["sequence"] for item in records) == [1, 2, 3, 4, 5]


def test_trace_path_pointing_at_a_regular_file_disables_tracing(tmp_path, caplog) -> None:
    """A pre-existing single-file JSONL trace must not crash the process."""

    trace_path = tmp_path / "old-trace.jsonl"
    trace_path.write_text('{"sequence": 1}\n')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client, trace_path=trace_path)
            return await adapter.complete(
                ChatCompletionRequest(
                    model="m", messages=[ChatMessage(role="user", content="hi")]))

    with caplog.at_level("WARNING"):
        result = asyncio.run(run())

    assert result == "ok"
    # The old file is left untouched -- no migration, no crash.
    assert trace_path.read_text() == '{"sequence": 1}\n'


def test_trace_is_off_by_default() -> None:
    """No trace path configured means no file handling at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://x", "k", "m", client=client)
            assert adapter.trace_path is None
            return await adapter.complete(
                ChatCompletionRequest(
                    model="m", messages=[ChatMessage(role="user", content="hi")]))

    assert asyncio.run(run()) == "ok"
