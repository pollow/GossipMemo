"""Tests for the single-permit, strict-priority provider gate in llm.py."""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import ContinuityReasoningResult
from gossipmemo.priority import TIER_BACKGROUND, TIER_FOREGROUND
from gossipmemo.transport import ChatMessage, ProviderGate, structured


async def _structured_call(adapter, system_prompt, user_prompt, result_type):
    """Reproduce the deleted `OpenAICompatibleAdapter._structured_call` seam.

    Any `LlmTransport` can drive `structured()` directly; this local helper
    keeps these gate tests focused on one system/user prompt pair without
    routing through a specific reasoner's chunking.
    """

    _, result = await structured(
        adapter,
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        result_type,
        tier=TIER_BACKGROUND,
        label=None,
    )
    return result


def test_tier_one_acquires_ahead_of_already_waiting_tier_three() -> None:
    """A foreground caller jumps a background caller already waiting."""

    async def run() -> list[str]:
        gate = ProviderGate()
        order: list[str] = []

        # Hold the single permit so both callers below have to queue.
        await gate.acquire(TIER_BACKGROUND, "holder")

        async def waiter(tier: int, label: str) -> None:
            await gate.acquire(tier, label)
            order.append(label)
            gate.release()

        background_waiter = asyncio.create_task(
            waiter(TIER_BACKGROUND, "background"))
        # let it enqueue before the foreground caller arrives
        await asyncio.sleep(0)
        foreground_waiter = asyncio.create_task(
            waiter(TIER_FOREGROUND, "foreground"))
        await asyncio.sleep(0)

        gate.release()  # frees the holder; strict priority must pick foreground next
        await asyncio.gather(background_waiter, foreground_waiter)
        return order

    assert asyncio.run(run()) == ["foreground", "background"]


def test_fifo_order_preserved_within_one_tier() -> None:
    """Same-tier waiters are served in submission order, no reordering."""

    async def run() -> list[str]:
        gate = ProviderGate()
        order: list[str] = []
        await gate.acquire(TIER_BACKGROUND, "holder")

        async def waiter(label: str) -> None:
            await gate.acquire(TIER_BACKGROUND, label)
            order.append(label)
            gate.release()

        tasks = []
        for label in ("first", "second", "third"):
            tasks.append(asyncio.create_task(waiter(label)))
            await asyncio.sleep(0)  # preserve enqueue order

        gate.release()
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(run()) == ["first", "second", "third"]


def test_shared_backoff_deadline_blocks_other_callers_acquisition() -> None:
    """A published Retry-After deadline holds off a fresh, unrelated acquire."""

    async def run() -> float:
        gate = ProviderGate()
        loop = asyncio.get_running_loop()
        gate.mark_unavailable_until(loop.time() + 0.15)
        started = loop.time()
        await gate.acquire(TIER_BACKGROUND, "waiter")
        elapsed = loop.time() - started
        gate.release()
        return elapsed

    elapsed = asyncio.run(run())
    assert elapsed >= 0.13


def test_adapter_publishes_shared_backoff_deadline_on_429_retry_after() -> None:
    """`_chat_messages` publishes the provider's Retry-After to the gate."""

    responses = [429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = responses.pop(0)
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "0.05"}, json={"error": "slow down"})
        body = {"text": "ok", "related_person_ids": [],
                "through_message_id": "m"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> list[float]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                retry_base_seconds=0.05, retry_max_seconds=0.05, max_retries=2,
            )
            recorded: list[float] = []
            original = adapter.gate.mark_unavailable_until

            def spy(deadline: float) -> None:
                recorded.append(deadline)
                original(deadline)

            # type: ignore[method-assign]
            adapter.gate.mark_unavailable_until = spy
            result = await _structured_call(adapter, "sys", "call", ContinuityReasoningResult)
            assert result.text == "ok"
            return recorded

    recorded = asyncio.run(run())
    assert len(recorded) == 1


def test_semantic_retry_backoff_does_not_block_other_caller() -> None:
    """Malformed-output retries in `_structured_messages` sit outside the gate."""

    attempts = {"a": 0}
    timeline: list[tuple[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompt = str(payload["messages"])
        now = time.monotonic()
        if "CALL-A" in prompt:
            attempts["a"] += 1
            timeline.append(("a", now))
            if attempts["a"] == 1:
                # Not valid JSON: forces the semantic retry inside
                # `_structured_messages`, which lives outside `_chat_messages`
                # and therefore must not hold the provider gate.
                return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
            body = {"text": "ok-a", "related_person_ids": [],
                    "through_message_id": "m"}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})
        timeline.append(("b", now))
        body = {"text": "ok-b", "related_person_ids": [],
                "through_message_id": "m"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    async def run() -> tuple[str, str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://x", "k", "m", client=client,
                retry_base_seconds=0.2, retry_max_seconds=0.2, max_retries=3,
            )
            task_a = asyncio.create_task(
                _structured_call(adapter, "sys-a", "CALL-A",
                                 ContinuityReasoningResult)
            )
            await asyncio.sleep(0.02)  # let A's first (malformed) attempt land
            task_b = asyncio.create_task(
                _structured_call(adapter, "sys-b", "call-b",
                                 ContinuityReasoningResult)
            )
            result_b = await task_b
            result_a = await task_a
            return result_a.text, result_b.text

    result_a, result_b = asyncio.run(run())
    assert (result_a, result_b) == ("ok-a", "ok-b")

    a_times = [t for label, t in timeline if label == "a"]
    b_times = [t for label, t in timeline if label == "b"]
    assert len(a_times) == 2
    assert len(b_times) == 1
    # B's only request must land strictly between A's first (malformed) and
    # second (retried) attempts -- proof B was never blocked by A's sleep.
    assert a_times[0] < b_times[0] < a_times[1]
