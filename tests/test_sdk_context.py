import asyncio

import httpx

from gossipmemo_client import AsyncGossipMemo, GossipMemo
from integrations.hermes.gossipmemo import GossipMemoMemoryProvider


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/context"):
        return httpx.Response(200, json={"version": "v1", "people": []})
    assert request.url.path.endswith("/turns")
    payload = request.read()
    assert b'"author":"user"' in payload
    assert b'"idempotency_key":"stable"' in payload
    return httpx.Response(202, json={"status": "accepted", "message_id": "m"})


def test_sync_context_and_turn_payload_validation():
    client = GossipMemo("http://test", client=httpx.Client(transport=httpx.MockTransport(_handler)))
    assert client.context()["version"] == "v1"
    assert client.turn(" hello ", idempotency_key="stable")["message_id"] == "m"
    try:
        client.prepare_turn({"author": "assistant", "content": "bad"})
    except ValueError:
        pass
    else:
        raise AssertionError("assistant turn must be normalized/rejected")


def test_async_context_and_turn_payload_validation():
    async def run():
        client = AsyncGossipMemo(
            "http://test", client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)))
        assert (await client.context())["version"] == "v1"
        assert (await client.turn("hello", idempotency_key="stable"))["message_id"] == "m"
        await client.close()

    asyncio.run(run())


def test_hermes_formats_context_and_reuses_key_for_slow_turn():
    import threading

    calls = []
    ingested = []
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class Fake:
        def turn(self, message, **kwargs):
            calls.append((message, kwargs.copy()))
            if len(calls) == 1:
                started.set()
                assert release.wait(2)
                finished.set()
                return {"context_update": {"version": "v1", "user_model": {"profile_card": {"summary": "calm"}}, "continuity": {"text": "thread"}, "people": [{"id": "p1", "display_name": "Alice"}]}, "known_people": [{"id": "p1", "display_name": "Alice"}], "memory_recall": [{"content": "likes tea"}]}
            if len(calls) == 2:
                return {"context_update": None, "known_people": [], "memory_recall": []}
            raise RuntimeError("offline")

        def ingest(self, messages):
            ingested.append(messages)

        def close(self): pass

    provider = GossipMemoMemoryProvider(client_factory=lambda **_: Fake())
    provider.initialize("s")
    try:
        # The first request is still in flight after prefetch's bounded wait.
        first_result = []
        caller = threading.Thread(target=lambda: first_result.append(provider.prefetch("hi")))
        caller.start()
        assert started.wait(1)
        caller.join(1)
        assert first_result == [""]
        provider.sync_turn("hi", "done")
        provider._queue.join()
        assert ingested[0][0]["idempotency_key"] == calls[0][1]["idempotency_key"]
        release.set()
        assert finished.wait(2)

        # Force a new preparation so the second request proves the cached
        # version is sent and the old bundle survives a null update.
        with provider._prefetch_lock:
            provider._prefetch_cache.clear()
        text = provider.prefetch("second")
        assert "calm" in text and "thread" in text and text.count("Person: Alice") == 1
        assert calls[1][1]["context_version"] == "v1"

        # Preparation failures are non-fatal, but the pre-registered key is
        # still used by the asynchronous completed-turn ingest.
        with provider._prefetch_lock:
            provider._prefetch_cache.clear()
        assert provider.prefetch("third") == ""
        provider.sync_turn("third", "done third")
        provider._queue.join()
        assert ingested[-1][0]["idempotency_key"] == calls[2][1]["idempotency_key"]
        assert ingested[-1][1]["author"] == "assistant"
    finally:
        release.set()
        provider.shutdown()
