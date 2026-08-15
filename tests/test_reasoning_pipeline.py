from __future__ import annotations

import asyncio

import pytest

from gossipmemo.reasoning import FunctionStage, ReasoningPipeline


@pytest.mark.asyncio
async def test_pipeline_is_strictly_sequential_and_stops_on_failure():
    events: list[str] = []

    async def first(space: str) -> None:
        events.append(f"start-{space}")
        await asyncio.sleep(0)
        events.append("first-done")

    async def failed(_: str) -> None:
        events.append("second-start")
        raise RuntimeError("boom")

    pipeline = ReasoningPipeline((FunctionStage("one", first), FunctionStage("two", failed), FunctionStage("three", lambda _: asyncio.sleep(0))))
    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.run_until_caught_up("s")
    assert events == ["start-s", "first-done", "second-start"]


@pytest.mark.asyncio
async def test_world_daily_scan_deduplicates_pipeline_tasks_per_space(tmp_path, monkeypatch):
    from gossipmemo.store import SqliteWorldStore
    from gossipmemo.world import SocialMemoryWorld

    world = SocialMemoryWorld(SqliteWorldStore(tmp_path / "world.db"), object())
    monkeypatch.setattr(world.store, "stale_entities", lambda: ([("space", "p")], [], []))
    monkeypatch.setattr(world.store, "stale_coverage_spaces", lambda: [])
    world._schedule_all_stale()
    world._schedule_all_stale()
    assert sum(key == ("reasoning-pipeline", "space", "space") for key in world._scheduled) == 1
    world._stopping = True
    await asyncio.gather(*world._tasks, return_exceptions=True)


def test_context_budget_handles_cjk_json_and_reports_fit():
    from gossipmemo.context_budget import ContextBudget

    budget = ContextBudget(110, 20, 10)
    estimate = budget.estimate_request({"messages": [{"role": "user", "content": "你好" * 20}]})
    report = budget.report(estimate)
    assert report.fits
    assert budget.estimate_text("你好") > budget.estimate_text("hi")


def test_context_budget_rejects_unusable_values():
    from gossipmemo.context_budget import ContextBudget

    with pytest.raises(ValueError, match="usable"):
        ContextBudget(10, 8, 3)


@pytest.mark.asyncio
async def test_adapter_rejects_oversized_request_before_http_client():
    from gossipmemo.context_budget import ContextBudget
    from gossipmemo.llm import ChatMessage, OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter("http://example.test", "key", "model", context_budget=ContextBudget(100, 90, 5))
    called = False

    async def no_client():
        nonlocal called
        called = True
        raise AssertionError("HTTP client must not be created")

    adapter._get_client = no_client  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=r"estimated .* limit"):
        await adapter._chat_messages([ChatMessage(role="user", content="超大文本" * 100)], structured=False)
    assert not called
