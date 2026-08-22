"""Tests for the background embedding worker: batching, storage-side
discipline (no instruction prefix), degrade-not-crash on provider failure,
and the health-endpoint observability the design brief asks for.

All offline: the LLM side uses a minimal `LlmTransport` double (mirroring
`tests/test_turn.py`'s `_NoopModel`) and the embedding side uses
`tests/fakes_embedding.py`'s `FakeEmbeddingClient` -- no real network calls
except the probe-failure test, which deliberately points at a closed local
port to exercise the real async probe path without depending on any
external service.

A scheduled background task (`world._spawn`) only actually runs once the
event loop is given a chance to advance -- `world.stop()` sets `_stopping`
before awaiting outstanding tasks, and every loop here (`_embedding_loop`,
`AttemptLoop.run_until_caught_up`) checks that flag before doing its first
unit of work. So tests that care whether the worker actually drained
`pending_embeddings()` poll for that between `start()` and `stop()`, the
same way `tests/test_turn.py` polls for continuity to land.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from harness import SilentTransport

from gossipmemo import world as world_module
from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.models import ManualMemoryRequest, MessageInput, TurnRequest
from gossipmemo.store import SqliteWorldStore
from gossipmemo.world import SocialMemoryWorld
from tests.fakes_embedding import FakeEmbeddingClient


class _FlakyEmbeddingClient:
    """Wraps a `FakeEmbeddingClient`, raising on the first N calls before
    delegating -- simulates the embedding provider being briefly down."""

    def __init__(self, delegate: FakeEmbeddingClient, failures: int) -> None:
        self._delegate = delegate
        self._remaining_failures = failures
        self.calls = delegate.calls

    @property
    def model(self) -> str:
        return self._delegate.model

    @property
    def dim(self) -> int:
        return self._delegate.dim

    async def embed(self, texts, *, instruction=None):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("embedding provider unavailable")
        return await self._delegate.embed(texts, instruction=instruction)


def _store(tmp_path: Path) -> SqliteWorldStore:
    store = SqliteWorldStore(tmp_path / "world.db")
    store.initialize()
    return store


def _add_memories(store: SqliteWorldStore, count: int) -> None:
    for i in range(count):
        store.add_manual_memory(
            "s", ManualMemoryRequest(content=f"memory number {i}", about_user=True)
        )


async def _start_drain_stop(
    world: SocialMemoryWorld, store: SqliteWorldStore, model: str, dim: int,
    *, attempts: int = 300, delay: float = 0.01,
) -> None:
    """Start the world, poll until embedding has caught up (or give up
    after `attempts`), then stop it -- mirrors `test_turn.py`'s continuity
    polling pattern rather than assuming a task ran just because it was
    scheduled."""

    await world.start()
    for _ in range(attempts):
        if store.pending_embedding_count(model=model, dim=dim) == 0:
            break
        await asyncio.sleep(delay)
    await world.stop()


async def _start_stop(world: SocialMemoryWorld) -> None:
    await world.start()
    await asyncio.sleep(0)
    await world.stop()


# -- batching / storage-side discipline ---------------------------------


def test_worker_drains_pending_in_batches_with_no_instruction_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(world_module, "_EMBEDDING_BATCH_SIZE", 2)
    store = _store(tmp_path)
    _add_memories(store, 5)
    client = FakeEmbeddingClient(model="fake-embedding", dim=8)
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=client)

    asyncio.run(_start_drain_stop(world, store, client.model, client.dim))

    assert store.pending_embedding_count(model=client.model, dim=client.dim) == 0
    # 5 items at a batch size of 2 must take at least 3 requests.
    assert len(client.calls) >= 3
    total_texts = sum(len(texts) for texts, _instruction in client.calls)
    assert total_texts == 5
    for texts, instruction in client.calls:
        assert len(texts) <= 2
        # The hard discipline: storage-side calls never carry a query prefix.
        assert instruction is None


def test_worker_is_a_noop_with_nothing_pending(tmp_path):
    store = _store(tmp_path)
    client = FakeEmbeddingClient()
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=client)

    asyncio.run(_start_drain_stop(world, store, client.model, client.dim))

    assert client.calls == []


# -- provider failure resilience -----------------------------------------


def test_worker_survives_provider_exception_and_recovers_next_round(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(world_module, "_EMBEDDING_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(world_module, "_EMBEDDING_RETRY_MAX_SECONDS", 0.02)
    store = _store(tmp_path)
    _add_memories(store, 3)
    delegate = FakeEmbeddingClient()
    flaky = _FlakyEmbeddingClient(delegate, failures=2)
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=flaky)

    with caplog.at_level(logging.ERROR, logger="gossipmemo.world"):
        asyncio.run(_start_drain_stop(world, store, flaky.model, flaky.dim))

    # It did not crash the world, and the pending rows eventually got
    # embedded once the simulated outage ended.
    assert store.pending_embedding_count(model=flaky.model, dim=flaky.dim) == 0
    assert any("embedding_batch_failed" in record.message for record in caplog.records)


def test_embedding_failure_never_reaches_the_turn_path(tmp_path, monkeypatch):
    monkeypatch.setattr(world_module, "_EMBEDDING_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(world_module, "_EMBEDDING_RETRY_MAX_SECONDS", 0.02)
    store = _store(tmp_path)

    class _AlwaysFailsEmbeddingClient:
        model = "fake-embedding"
        dim = 8

        async def embed(self, texts, *, instruction=None):
            raise RuntimeError("embedding provider unavailable")

    world = SocialMemoryWorld(
        store, SilentTransport(), embedding_client=_AlwaysFailsEmbeddingClient()
    )

    async def scenario():
        await world.start()
        try:
            response = await world.turn(
                "s", TurnRequest(messages=[MessageInput(author="user", content="hello")])
            )
            assert response.status == "accepted"
        finally:
            await world.stop()

    asyncio.run(scenario())


# -- startup wiring / degradation ----------------------------------------


def test_unconfigured_embedding_starts_cleanly_and_health_reflects_it(tmp_path):
    store = _store(tmp_path)
    world = SocialMemoryWorld(store, SilentTransport())

    asyncio.run(_start_stop(world))

    health = world.health()
    assert health.embedding_enabled is False
    assert health.embedding_pending == 0


def test_health_reports_enabled_and_pending_count(tmp_path):
    store = _store(tmp_path)
    _add_memories(store, 2)
    client = FakeEmbeddingClient()
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=client)

    health_before = world.health()
    assert health_before.embedding_enabled is True
    assert health_before.embedding_pending == 2

    asyncio.run(_start_drain_stop(world, store, client.model, client.dim))

    health_after = world.health()
    assert health_after.embedding_pending == 0


def test_health_endpoint_ok_when_embedding_unconfigured(tmp_path):
    import httpx

    async def scenario():
        settings = Settings(
            database_path=tmp_path / "app.db",
            llm_base_url="http://llm.test/v1", llm_api_key="key", llm_model="model",
        )
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")
                assert response.status_code == 200
                payload = response.json()
                assert payload["embedding_enabled"] is False
                assert payload["embedding_pending"] == 0

    asyncio.run(scenario())


def test_probe_failure_disables_subsystem_but_still_starts(tmp_path):
    """`GOSSIPMEMO_EMBEDDING_MODEL` set but the provider unreachable, and no
    `GOSSIPMEMO_EMBEDDING_DIM` fallback configured -- dimension resolution
    fails, and startup must still succeed with embedding simply off."""

    settings = Settings(
        database_path=tmp_path / "app.db",
        llm_base_url="http://llm.test/v1", llm_api_key="key", llm_model="model",
        embedding_base_url="http://127.0.0.1:1/v1", embedding_model="probe-model",
        # A closed local port is expected to refuse instantly, but the
        # sandbox this test may run in can instead let the connect attempt
        # hang until timeout -- keep that bounded rather than inheriting
        # the 120s default.
        llm_timeout_seconds=2.0,
    )
    store = _store(tmp_path)
    world = SocialMemoryWorld(store, SilentTransport(), settings=settings)

    asyncio.run(_start_stop(world))

    health = world.health()
    assert health.embedding_enabled is False
    assert health.embedding_pending == 0


# -- backfill on startup --------------------------------------------------


def test_startup_backfill_processes_pre_existing_rows(tmp_path):
    store = _store(tmp_path)
    _add_memories(store, 4)
    client = FakeEmbeddingClient()
    # Settings-driven creation is exercised elsewhere; here the client is
    # injected directly (as tests do), but the point under test is that
    # `start()` schedules a full backfill over rows that already existed
    # before the process came up, not just newly-arriving ones.
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=client)

    assert store.pending_embedding_count(model=client.model, dim=client.dim) == 4

    asyncio.run(_start_drain_stop(world, store, client.model, client.dim))

    assert store.pending_embedding_count(model=client.model, dim=client.dim) == 0
    assert len(client.calls) >= 1


# -- shutdown resource cleanup --------------------------------------------


class _ClosableEmbeddingClient:
    """An `EmbeddingClient` that *does* implement `aclose` -- unlike
    `FakeEmbeddingClient`, which deliberately doesn't (the Protocol makes
    it optional). Used to assert `stop()` actually calls it."""

    model = "fake-embedding"
    dim = 8

    def __init__(self) -> None:
        self.closed = False

    async def embed(self, texts, *, instruction=None):
        return [[0.0] * self.dim for _ in texts]

    async def aclose(self) -> None:
        self.closed = True


def test_stop_closes_an_embedding_client_that_declares_aclose(tmp_path):
    store = _store(tmp_path)
    client = _ClosableEmbeddingClient()
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=client)

    asyncio.run(_start_stop(world))

    assert client.closed is True


def test_stop_does_not_choke_on_a_client_with_no_aclose(tmp_path):
    """`FakeEmbeddingClient` has no `aclose` -- `stop()` must treat that as
    nothing to close, not as an error."""

    store = _store(tmp_path)
    client = FakeEmbeddingClient()
    assert not hasattr(client, "aclose")
    world = SocialMemoryWorld(store, SilentTransport(), embedding_client=client)

    asyncio.run(_start_stop(world))  # must not raise
