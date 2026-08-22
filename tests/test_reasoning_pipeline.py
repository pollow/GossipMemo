from __future__ import annotations

import asyncio

import pytest

from gossipmemo.reasoners import AttemptLoop
from gossipmemo.reasoning import ReasoningPipeline


class RecordingReasoner(AttemptLoop):
    """A reasoner that reports `attempts` units of work, then stops."""

    def __init__(self, name: str, events: list[str], attempts: int = 1, boom: bool = False) -> None:
        self.name = name
        self._events = events
        self._remaining = attempts
        self._boom = boom

    async def _attempt(self, space_id: str) -> bool:
        self._events.append(f"{self.name}-{space_id}")
        await asyncio.sleep(0)
        if self._boom:
            raise RuntimeError("boom")
        self._remaining -= 1
        return self._remaining > 0


@pytest.mark.asyncio
async def test_pipeline_is_strictly_sequential_and_stops_on_failure():
    events: list[str] = []
    pipeline = ReasoningPipeline(
        (
            RecordingReasoner("one", events),
            RecordingReasoner("two", events, boom=True),
            RecordingReasoner("three", events),
        )
    )
    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.run_until_caught_up("s")
    assert events == ["one-s", "two-s"]


@pytest.mark.asyncio
async def test_pipeline_runs_each_reasoner_to_catch_up():
    events: list[str] = []
    pipeline = ReasoningPipeline(
        (RecordingReasoner("one", events, attempts=3), RecordingReasoner("two", events))
    )
    await pipeline.run_until_caught_up("s")
    assert events == ["one-s", "one-s", "one-s", "two-s"]


@pytest.mark.asyncio
async def test_catch_up_stops_between_attempts_when_asked():
    events: list[str] = []
    reasoner = RecordingReasoner("one", events, attempts=5)
    # The reasoner still has work; the predicate is re-checked before each
    # unit, so it stops mid-catch-up rather than draining.
    await reasoner.run_until_caught_up("s", lambda: not events)
    assert events == ["one-s"]


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


def test_disabled_reasoners_are_dropped_from_the_pipeline(tmp_path):
    from harness import NoopTransport, settings

    from gossipmemo.config import ConfigurationError
    from gossipmemo.store import SqliteWorldStore
    from gossipmemo.world import SocialMemoryWorld

    store = SqliteWorldStore(tmp_path / "world.db")
    world = SocialMemoryWorld(
        store, NoopTransport(),
        settings=settings(tmp_path, disabled_reasoners=("user_model", "coverage")),
    )
    names = [reasoner.name for reasoner in world.reasoning_pipeline._reasoners]
    assert names == ["person", "relationship", "learning_goals"]

    with pytest.raises(ConfigurationError, match="nope"):
        SocialMemoryWorld(
            SqliteWorldStore(tmp_path / "other.db"), NoopTransport(),
            settings=settings(tmp_path, disabled_reasoners=("nope",)),
        )
