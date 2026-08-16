from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest

from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.llm import ProviderGate
from gossipmemo.models import ManualMemoryRequest
from gossipmemo.store import PersonMergeError, SqliteWorldStore
from gossipmemo.world import SocialMemoryWorld
from gossipmemo_client import AsyncGossipMemo, GossipMemo
from integrations.hermes.gossipmemo import GossipMemoMemoryProvider


def rows(store: SqliteWorldStore, query: str, params: tuple = ()) -> list[sqlite3.Row]:
    with store._connect() as connection:
        return connection.execute(query, params).fetchall()


def test_merge_reconnects_evidence_aliases_and_continuity(tmp_path):
    store = SqliteWorldStore(tmp_path / "merge.db")
    store.initialize()
    store.add_manual_memory(
        "s", ManualMemoryRequest(content="Alex fact", people=["Alex Wang"])
    )
    store.add_manual_memory("s", ManualMemoryRequest(content="AW fact", people=["AW"]))
    people = rows(
        store,
        "SELECT id, display_name FROM people "
        "WHERE space_id = 's' ORDER BY display_name",
    )
    target, source = people[1]["id"], people[0]["id"]
    store.ensure_space("s")
    with store._connect() as connection:
        connection.execute(
            "UPDATE continuities SET related_person_ids = ? WHERE space_id = 's'",
            (f'["{source}", "{target}"]',),
        )
        before = connection.execute(
            "SELECT content FROM memories ORDER BY created_at"
        ).fetchall()
    result = store.merge_person("s", source, target)
    assert result["status"] == "merged"
    assert rows(
        store,
        "SELECT person_id FROM person_aliases WHERE normalized_value = 'aw'",
    )[0]["person_id"] == target
    assert not rows(store, "SELECT 1 FROM memory_people WHERE person_id = ?", (source,))
    assert rows(
        store,
        "SELECT status, merged_into_person_id FROM people WHERE id = ?",
        (source,),
    )[0]["merged_into_person_id"] == target
    assert rows(
        store,
        "SELECT related_person_ids FROM continuities WHERE space_id = 's'",
    )[0]["related_person_ids"] == f'["{target}"]'
    assert [
        row["content"]
        for row in rows(store, "SELECT content FROM memories ORDER BY created_at")
    ] == [row["content"] for row in before]
    assert store.merge_person("s", source, target)["status"] == "merged"


def test_merge_conflicts_are_explicit(tmp_path):
    store = SqliteWorldStore(tmp_path / "merge-conflict.db")
    store.initialize()
    store.add_manual_memory("s", ManualMemoryRequest(content="a", people=["A"]))
    store.add_manual_memory("s", ManualMemoryRequest(content="b", people=["B"]))
    ids = [
        row["id"]
        for row in rows(
            store,
            "SELECT id FROM people WHERE space_id = 's' ORDER BY display_name",
        )
    ]
    with pytest.raises(PersonMergeError):
        store.merge_person("missing", ids[0], ids[1])
    with pytest.raises(PersonMergeError) as error:
        store.merge_person("s", ids[0], ids[0])
    assert error.value.conflict


def test_merge_rewires_relationships_and_removes_self_relation(tmp_path):
    store = SqliteWorldStore(tmp_path / "relationship-merge.db")
    store.initialize()
    for name in ("Alex", "AW", "Bob"):
        store.add_manual_memory("s", ManualMemoryRequest(content=name, people=[name]))
    people = {
        row["display_name"]: row["id"]
        for row in rows(store, "SELECT id, display_name FROM people WHERE space_id = 's'")
    }
    alex, aw, bob = people["Alex"], people["AW"], people["Bob"]
    memories = rows(store, "SELECT id FROM memories WHERE space_id = 's'")
    now = "2026-01-01T00:00:00+00:00"
    with store._connect() as connection:
        relationships = (
            ("r_self", aw, alex),
            ("r_aw_bob", aw, bob),
            ("r_alex_bob", alex, bob),
        )
        for relation_id, left, right in relationships:
            a_id, b_id = sorted((left, right))
            connection.execute(
                "INSERT INTO relationships("
                "id, space_id, person_a_id, person_b_id, facets, closeness, "
                "tone, status, summary, profile_source_updated_at, "
                "profile_updated_at, created_at, updated_at"
                ") VALUES (?, 's', ?, ?, '[\"friend\"]', 'close', 'warm', "
                "'active', ?, ?, ?, ?, ?)",
                (relation_id, a_id, b_id, relation_id, now, now, now, now),
            )
        connection.execute(
            "INSERT INTO memory_relationships(memory_id, relationship_id) VALUES (?, 'r_aw_bob')",
            (memories[0]["id"],),
        )
        connection.execute(
            "INSERT INTO memory_relationships(memory_id, relationship_id) "
            "VALUES (?, 'r_alex_bob')",
            (memories[1]["id"],),
        )
    result = store.merge_person("s", aw, alex)
    assert "r_alex_bob" in result["affected_relationship_ids"]
    relations = rows(
        store,
        "SELECT id, person_a_id, person_b_id, facets, closeness, tone, "
        "summary, status, profile_source_updated_at, profile_updated_at "
        "FROM relationships WHERE space_id = 's'",
    )
    assert len(relations) == 1
    assert relations[0]["id"] == "r_alex_bob"
    assert relations[0]["facets"] == "[]"
    assert relations[0]["closeness"] is None
    assert relations[0]["tone"] is None
    assert relations[0]["summary"] == ""
    assert relations[0]["status"] == "unknown"
    assert relations[0]["profile_source_updated_at"] is None
    assert relations[0]["profile_updated_at"] is None
    assert not rows(store, "SELECT 1 FROM relationships WHERE id = 'r_self'")
    assert not rows(store, "SELECT 1 FROM relationships WHERE id = 'r_aw_bob'")
    relationship_memories = rows(
        store,
        "SELECT memory_id FROM memory_relationships "
        "WHERE relationship_id = 'r_alex_bob'",
    )
    assert len(relationship_memories) == 2


def test_merge_http_endpoint_and_status_mapping(tmp_path: Path):
    async def scenario() -> None:
        store = SqliteWorldStore(tmp_path / "merge-http.db")
        store.initialize()
        store.add_manual_memory("s", ManualMemoryRequest(content="A", people=["A"]))
        store.add_manual_memory("s", ManualMemoryRequest(content="B", people=["B"]))
        ids = [
            row["id"]
            for row in rows(
                store,
                "SELECT id FROM people "
                "WHERE space_id = 's' ORDER BY display_name",
            )
        ]
        world = SocialMemoryWorld(store, _NoopModel())
        settings = Settings(
            database_path=store.path,
            llm_base_url="http://llm.test/v1",
            llm_api_key="key",
            llm_model="model",
        )
        app = create_app(settings, world)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/v1/spaces/s/people/{ids[0]}/merge",
                    json={"target_person_id": ids[1]},
                )
                assert response.status_code == 200
                assert response.json()["status"] == "merged"
                missing = await client.post(
                    "/v1/spaces/s/people/missing/merge",
                    json={"target_person_id": ids[1]},
                )
                assert missing.status_code == 404
                conflict = await client.post(
                    f"/v1/spaces/s/people/{ids[1]}/merge",
                    json={"target_person_id": ids[1]},
                )
                assert conflict.status_code == 409

    asyncio.run(scenario())


class _NoopModel:
    configured = False
    gate = ProviderGate()

    async def aclose(self):
        return None


def test_sdk_merge_methods_send_source_target_payloads():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "status": "merged",
                "source_person_id": "s",
                "target_person_id": "t",
            },
        )

    with GossipMemo("http://test", transport=httpx.MockTransport(handler)) as client:
        assert client.merge_person("s", "t")["status"] == "merged"

    async def scenario() -> None:
        async with AsyncGossipMemo(
            "http://test", transport=httpx.MockTransport(handler)
        ) as client:
            assert (await client.merge_person("s", "t"))["status"] == "merged"

    asyncio.run(scenario())
    assert [request.url.path for request in seen] == ["/v1/spaces/personal/people/s/merge"] * 2
    assert all(request.content == b'{"target_person_id":"t"}' for request in seen)


def test_hermes_merge_tool_requires_ids_and_calls_client():
    class FakeClient:
        def merge_person(self, source: str, target: str):
            return {"source_person_id": source, "target_person_id": target, "status": "merged"}

        def close(self):
            return None

    provider = GossipMemoMemoryProvider(client_factory=lambda **_: FakeClient())
    provider.initialize("session")
    try:
        schema = next(
            item
            for item in provider.get_tool_schemas()
            if item["name"] == "gossipmemo_merge_people"
        )
        assert schema["parameters"]["required"] == ["source_person_id", "target_person_id"]
        assert "confirmed" in schema["description"]
        result = provider.handle_tool_call(
            "gossipmemo_merge_people",
            {"source_person_id": "s", "target_person_id": "t"},
        )
        assert '"status": "merged"' in result
    finally:
        provider.shutdown()
