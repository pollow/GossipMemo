import asyncio
import json
import re
from pathlib import Path

import httpx

from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.context_budget import ContextBudget
from gossipmemo.models import (
    ManualMemoryRequest,
    MessageInput,
    TurnRequest,
)
from gossipmemo.store import SqliteWorldStore
from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
from gossipmemo.world import SocialMemoryWorld


class _NoopModel:
    """A minimal `LlmTransport` double: extraction/continuity/coverage/goal
    stages are all told apart by prompt marker in `complete`, matching the
    pattern in `reasoners/owner.py` and friends -- see `tests/test_features.py`'s
    `FakeModel` for the fuller version of this same shape.
    """

    configured = False
    gate = ProviderGate()
    context_budget = ContextBudget()
    retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)
    user_name = "CurrentUser"
    extraction_policy = "balanced"

    async def aclose(self):
        return None

    def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake",
            messages=list(messages),
            response_format={"type": "json_object"} if structured else None,
        )

    async def complete(self, request: ChatCompletionRequest) -> str:
        combined = " ".join(str(message.content) for message in request.messages)
        if "Rebuild compact cross-session continuity." in combined:
            # Messages are rendered as JSON objects, so the last "id" in
            # the prompt is the last message the request covers.
            ids = re.findall(r'"id": "([^"]*)"', combined)
            return json.dumps({"text": "rolling", "through_message_id": ids[-1] if ids else ""})
        return json.dumps({})


def test_turn_matches_longest_alias_and_recalls_user_memory(tmp_path: Path):
    store = SqliteWorldStore(tmp_path / "turn.db")
    store.initialize()
    store.add_manual_memory("s", ManualMemoryRequest(
        content="Alice likes tea", people=["Alice"], about_user=True))
    # Add an overlapping alias to ensure the shorter alias cannot win.
    with store._connect() as connection:
        person = connection.execute(
            "SELECT id FROM people WHERE display_name = 'Alice'").fetchone()["id"]
        connection.execute(
            "INSERT INTO person_aliases(id, space_id, person_id, value, normalized_value) "
            "VALUES ('a', 's', ?, 'Al', 'al')",
            (person,),
        )

    async def scenario():
        world = SocialMemoryWorld(store, _NoopModel(), extraction_batch_size=100)
        await world.start()
        try:
            response = await world.turn(
                "s", TurnRequest(messages=[MessageInput(author="user", content="ALICE likes tea")])
            )
            assert response.status == "accepted"
            assert [person.display_name for person in response.known_people] == ["Alice"]
            assert response.memory_recall[0].content == "Alice likes tea"
            assert store.unbatched_messages("s")
            same = await world.turn(
                "s", TurnRequest(messages=[MessageInput(author="user", content="another")],
                                 context_version=response.context_update.version)
            )
            assert same.context_update is None
        finally:
            await world.stop()

    asyncio.run(scenario())


def test_turn_accepts_when_context_read_fails(tmp_path: Path):
    store = SqliteWorldStore(tmp_path / "turn-fail.db")
    store.initialize()
    world = SocialMemoryWorld(store, _NoopModel(), extraction_batch_size=100)
    original = store._context_state
    # `turn_view` now shares one `_context_state` read across guidance and
    # the context bundle, so that shared read is the surface to fail here
    # (this used to patch `context_bundle` directly, back when the turn
    # path called it as an independent, separately-connected store call).
    store._context_state = lambda connection, space_id: (
        _ for _ in ()).throw(RuntimeError("offline"))

    async def scenario():
        response = await world.turn(
            "s", TurnRequest(messages=[MessageInput(author="user", content="hello")]))
        assert response.status == "accepted"
        assert response.context_status == "unavailable"
        assert store.pending_extractions()

    asyncio.run(scenario())
    store._context_state = original


def test_turn_schedules_continuity_and_batches_ending_in_assistant_skip_enrichment(tmp_path: Path):
    async def scenario():
        store = SqliteWorldStore(tmp_path / "continuity-turn.db")
        world = SocialMemoryWorld(
            store, _NoopModel(), extraction_batch_size=100, continuity_threshold=2
        )
        await world.start()
        try:
            await world.turn(
                "s", TurnRequest(messages=[MessageInput(author="user", content="one")]))
            await world.turn(
                "s", TurnRequest(messages=[MessageInput(author="user", content="two")]))
            for _ in range(50):
                if store.context_bundle("s").continuity.text == "rolling":
                    break
                await asyncio.sleep(0)
            assert store.context_bundle("s").continuity.text == "rolling"

            app = create_app(
                Settings(database_path=store.path, llm_base_url="http://llm.test/v1",
                         llm_api_key="key", llm_model="model"),
                world,
            )
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    # A batch whose last message is from the assistant is
                    # still accepted and persisted, but read-enrichment
                    # (alias matching, guidance, FTS recall, context
                    # version comparison) is skipped -- it is derived from
                    # the batch's last message, not a caller flag.
                    response = await client.post(
                        "/v1/spaces/s/turns",
                        json={"messages": [{"author": "assistant", "content": "no"}]},
                    )
                    assert response.status_code == 202
                    payload = response.json()
                    assert payload["known_people"] == []
                    assert payload["memory_recall"] == []
                    assert payload["context_update"] is None
                    assert payload["context_status"] == "available"

                    # A batch ending in a user message still enriches, even
                    # when an assistant message precedes it in the batch.
                    enriched = await client.post(
                        "/v1/spaces/s/turns",
                        json={
                            "messages": [
                                {"author": "assistant", "content": "no"},
                                {"author": "user", "content": "Alice likes tea"},
                            ]
                        },
                    )
                    assert enriched.status_code == 202
                    enriched_payload = enriched.json()
                    assert enriched_payload["context_status"] == "available"
                    assert enriched_payload["context_update"] is not None
        finally:
            await world.stop()

    asyncio.run(scenario())


def test_turn_guidance_is_activated_and_generic_version_is_stable(tmp_path: Path):
    store = SqliteWorldStore(tmp_path / "guidance.db")
    store.initialize()
    store.ensure_space("s")
    store.add_manual_memory("s", ManualMemoryRequest(content="Alice is a friend", people=["Alice"]))
    with store._connect() as connection:
        pid = connection.execute(
            "SELECT id FROM people WHERE display_name = 'Alice'").fetchone()["id"]
        connection.execute(
            "INSERT INTO hypotheses(id,space_id,owner_kind,owner_id,content,kind,confidence,"
            "created_at,updated_at) "
            "VALUES ('h','s','person',?,'Alice may move','impression','low','1','1')", (pid,))
        connection.execute(
            "INSERT INTO hypotheses(id,space_id,owner_kind,owner_id,content,kind,confidence,"
            "created_at,updated_at) "
            "VALUES ('new','s','person',?,'unrelated newest','impression','low','3','3')", (pid,))
        connection.execute(
            "INSERT INTO hypotheses(id,space_id,owner_kind,owner_id,content,kind,confidence,"
            "created_at,updated_at) "
            "VALUES ('cjk','s','person',?,'旅行计划未定','impression','low','0','0')", (pid,))
        connection.execute(
            "INSERT INTO hypotheses(id,space_id,owner_kind,owner_id,content,kind,confidence,"
            "status,created_at,updated_at) "
            "VALUES ('closed','s','person',?,'旅行已确认','impression','low','rejected','4','4')",
            (pid,))
        connection.execute(
            "INSERT INTO learning_goals(id,space_id,prompt,rationale,entry_ids,status,"
            "created_at,updated_at) "
            "VALUES ('deferred','s','旅行方向','context','[]','deferred','5','5')")
        connection.execute(
            "INSERT INTO learning_goals(id,space_id,prompt,rationale,entry_ids,created_at,"
            "updated_at) VALUES ('g','s','Learn Alice plans','context','[]','2','2')")
    generic = store.context_bundle("s")
    assert [item.id for item in generic.guidance.items] == ["g"]
    assert store.context_bundle("s").version == generic.version
    selected = store.guidance_bundle("s", [pid], "Alice")
    assert [item.id for item in selected.items] == ["h", "g"]
    assert store.guidance_bundle("s", [pid], "旅行").items[0].id == "cjk"

    async def scenario():
        world = SocialMemoryWorld(store, _NoopModel(), extraction_batch_size=100)
        response = await world.turn(
            "s", TurnRequest(messages=[MessageInput(author="user", content="Alice?")]))
        assert any(item.id == "h" for item in response.guidance.items)
    before = store.context_bundle("s").version
    asyncio.run(scenario())
    assert store.context_bundle("s").version == before
