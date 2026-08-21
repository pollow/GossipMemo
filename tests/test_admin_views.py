from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.models import ManualMemoryRequest, MessageInput, SourceRef
from gossipmemo.store import SqliteWorldStore
from gossipmemo.store.policy import new_id, now_iso
from gossipmemo.world import SocialMemoryWorld

ADMIN_PASSWORD = "correct-horse-battery-staple"
XSS = "<script>alert(1)</script>"


class _NoopModel:
    """Minimal `LlmTransport` double -- admin routes never call the model."""

    configured = False

    async def aclose(self):
        return None


def _settings(tmp_path: Path, *, admin_password: str = ADMIN_PASSWORD) -> Settings:
    return Settings(
        database_path=tmp_path / "world.db",
        llm_base_url="http://llm.test/v1",
        llm_api_key="key",
        llm_model="model",
        admin_password=admin_password,
    )


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _seed_space(store: SqliteWorldStore, space_id: str, name: str) -> dict:
    """A small, realistic fixture: a space with messages spanning one
    extraction batch, a manual memory linked to an XSS-laden person name,
    a batch-derived memory that derives from it, a retracted memory to
    exercise the state filter, and a relationship between two people.
    """

    store.ensure_space(space_id, name)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    message_ids = store.record_messages(
        space_id,
        [
            MessageInput(
                author="user",
                content=f"My friend {XSS} likes coffee",
                occurred_at=base,
                source=SourceRef(provider="test"),
            ),
            MessageInput(
                author="assistant",
                content="Noted.",
                occurred_at=base + timedelta(minutes=1),
                source=SourceRef(provider="test"),
            ),
        ],
    )

    batch_id = new_id("batch")
    now = now_iso()
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO extraction_batches(id, space_id, created_at, completed_at) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, space_id, now, now),
        )
        connection.executemany(
            "UPDATE messages SET extraction_batch_id = ?, extraction_state = 'completed', "
            "extracted_at = ? WHERE id = ?",
            [(batch_id, now, message_id) for message_id in message_ids],
        )

    manual_memory_id = store.add_manual_memory(
        space_id,
        ManualMemoryRequest(
            content=f"Alice {XSS} likes coffee",
            kind="fact",
            people=["Alice"],
            about_user=False,
        ),
    )

    derived_memory_id = new_id("memory")
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO memories(
                id, space_id, content, kind, basis, about_user, status,
                source_batch_id, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, 'plan', 'inferred', 1, 'active', ?, 'llm', ?, ?)
            """,
            (derived_memory_id, space_id, "User is planning a trip", batch_id, now, now),
        )
        connection.execute(
            "INSERT INTO memory_derivations(derived_memory_id, source_memory_id, "
            "derivation_role) VALUES (?, ?, 'support')",
            (derived_memory_id, manual_memory_id),
        )

    retracted_memory_id = new_id("memory")
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO memories(
                id, space_id, content, kind, basis, about_user, status,
                created_by, created_at, updated_at
            ) VALUES (?, ?, ?, 'event', 'explicit', 0, 'retracted', 'llm', ?, ?)
            """,
            (retracted_memory_id, space_id, "Old plan, no longer true", now, now),
        )

    bob_id = new_id("person")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO people(id, space_id, display_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (bob_id, space_id, "Bob", now, now),
        )
        alice_id = connection.execute(
            "SELECT person_id FROM memory_people WHERE memory_id = ?", (manual_memory_id,)
        ).fetchone()["person_id"]
        a_id, b_id = sorted((alice_id, bob_id))
        relationship_id = new_id("relationship")
        connection.execute(
            "INSERT INTO relationships(id, space_id, person_a_id, person_b_id, "
            "summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (relationship_id, space_id, a_id, b_id, "friends", now, now),
        )
        connection.execute(
            "INSERT INTO memory_relationships(memory_id, relationship_id, role) "
            "VALUES (?, ?, 'about')",
            (manual_memory_id, relationship_id),
        )

    return {
        "manual_memory_id": manual_memory_id,
        "derived_memory_id": derived_memory_id,
        "retracted_memory_id": retracted_memory_id,
        "batch_id": batch_id,
        "message_ids": message_ids,
        "alice_id": alice_id,
        "bob_id": bob_id,
    }


async def _login(client: httpx.AsyncClient, password: str = ADMIN_PASSWORD) -> None:
    response = await client.post(
        "/admin/login", data={"password": password}, follow_redirects=False
    )
    assert response.status_code == 303


async def _run(tmp_path: Path, scenario, *, seed_spaces: list[str] | None = None):
    store = SqliteWorldStore(tmp_path / "world.db")
    world = SocialMemoryWorld(store, _NoopModel())
    app = create_app(_settings(tmp_path), world)
    fixtures: dict[str, dict] = {}
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            for space_id in seed_spaces or []:
                fixtures[space_id] = _seed_space(store, space_id, f"Space {space_id}")
            await _login(client)
            await scenario(client, fixtures)


def test_single_space_redirects_from_admin_root(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/spaces/space1"

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_two_spaces_renders_space_list(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin", follow_redirects=False)
        assert response.status_code == 200
        assert "Space space1" in response.text
        assert "Space space2" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1", "space2"]))


def test_space_list_route_also_works_directly(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces")
        assert response.status_code == 200
        assert "Space space1" in response.text
        assert "Space space2" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1", "space2"]))


def test_space_overview_shows_counts_user_model_and_continuity(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1")
        assert response.status_code == 200
        assert "Messages: 2" in response.text
        assert "Memories: 3" in response.text
        assert "People: 2" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_unknown_space_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/does-not-exist")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_messages_view_lists_author_and_batch(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/messages")
        assert response.status_code == 200
        assert "user" in response.text
        assert "assistant" in response.text
        assert fixtures["space1"]["batch_id"] in response.text
        assert "yes" in response.text  # extracted marker

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_messages_pagination_first_page_last_page_and_out_of_range(tmp_path: Path):
    async def scenario(client, fixtures):
        first = await client.get("/admin/spaces/space1/messages?offset=0&limit=1")
        assert first.status_code == 200
        assert "1-1 of 2" in first.text

        last = await client.get("/admin/spaces/space1/messages?offset=1&limit=1")
        assert last.status_code == 200
        assert "2-2 of 2" in last.text

        out_of_range = await client.get("/admin/spaces/space1/messages?offset=1000&limit=1")
        assert out_of_range.status_code == 200
        assert "No results" in out_of_range.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_messages_pagination_rejects_malformed_offset_without_error(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/messages?offset=not-a-number")
        assert response.status_code == 200

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memories_view_lists_all_three_states_unfiltered(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/memories")
        assert response.status_code == 200
        assert "3 of 3" in response.text or "1-3 of 3" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memories_filter_by_state_actually_filters(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/memories?state=retracted")
        assert response.status_code == 200
        assert "1-1 of 1" in response.text
        assert "Old plan" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memories_filter_by_kind_and_about_user(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get(
            "/admin/spaces/space1/memories?kind=plan&about_user=1"
        )
        assert response.status_code == 200
        assert "1-1 of 1" in response.text
        assert "planning a trip" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memories_filter_rejects_unknown_values_without_crashing(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get(
            "/admin/spaces/space1/memories?state=DROP TABLE memories;--"
        )
        assert response.status_code == 200
        # Malformed filter is dropped -> falls back to unfiltered (3 memories).
        assert "1-3 of 3" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memory_detail_shows_people_derivations_and_source_messages(tmp_path: Path):
    async def scenario(client, fixtures):
        manual_id = fixtures["space1"]["manual_memory_id"]
        derived_id = fixtures["space1"]["derived_memory_id"]

        manual_page = await client.get(f"/admin/spaces/space1/memories/{manual_id}")
        assert manual_page.status_code == 200
        assert "Alice" in manual_page.text
        assert "friends" in manual_page.text
        assert derived_id in manual_page.text  # linked in "Derives"

        derived_page = await client.get(f"/admin/spaces/space1/memories/{derived_id}")
        assert derived_page.status_code == 200
        assert manual_id in derived_page.text  # linked in "Derived from"
        assert "My friend" in derived_page.text  # source message content
        assert "Noted." in derived_page.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memory_detail_unknown_id_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/memories/does-not-exist")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_person_name_and_memory_content_are_escaped_everywhere(tmp_path: Path):
    async def scenario(client, fixtures):
        manual_id = fixtures["space1"]["manual_memory_id"]

        memories_list = await client.get("/admin/spaces/space1/memories")
        assert "<script>" not in memories_list.text
        assert "&lt;script&gt;" in memories_list.text

        detail = await client.get(f"/admin/spaces/space1/memories/{manual_id}")
        assert "<script>" not in detail.text
        assert "&lt;script&gt;" in detail.text

        messages = await client.get("/admin/spaces/space1/messages")
        assert "<script>" not in messages.text
        assert "&lt;script&gt;" in messages.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


@pytest.mark.parametrize(
    "path",
    [
        "/admin/spaces",
        "/admin/spaces/space1",
        "/admin/spaces/space1/messages",
        "/admin/spaces/space1/memories",
    ],
)
def test_every_view_requires_a_session(tmp_path: Path, path):
    async def scenario():
        store = SqliteWorldStore(tmp_path / "world.db")
        world = SocialMemoryWorld(store, _NoopModel())
        app = create_app(_settings(tmp_path), world)
        async with app.router.lifespan_context(app):
            _seed_space(store, "space1", "Space space1")
            async with _client(app) as client:
                response = await client.get(path, follow_redirects=False)
                assert response.status_code == 303
                assert response.headers["location"] == "/admin/login"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "path",
    [
        "/admin/spaces",
        "/admin/spaces/space1",
        "/admin/spaces/space1/messages",
        "/admin/spaces/space1/memories",
    ],
)
def test_every_view_carries_csp_header(tmp_path: Path, path):
    async def scenario(client, fixtures):
        response = await client.get(path)
        assert response.headers.get("Content-Security-Policy")
        assert response.headers.get("X-Frame-Options") == "DENY"

    asyncio.run(_run(tmp_path, lambda client, fixtures: scenario(client, fixtures),
                     seed_spaces=["space1"]))


def test_user_model_card_is_pretty_printed(tmp_path: Path):
    """The stored card is compact JSON; the overview must indent it."""

    async def scenario(store, client):
        response = await client.get("/admin/spaces/space1")
        assert response.status_code == 200
        # `esc()` turns the JSON quotes into entities; the browser renders
        # them back, so assert on the escaped form the page actually serves.
        assert "&quot;summary&quot;: &quot;loves coffee&quot;" in response.text
        assert "\n    &quot;espresso&quot;" in response.text
        # The compact one-line form must be gone.
        assert "{&quot;summary&quot;" not in response.text

    async def run():
        store = SqliteWorldStore(tmp_path / "world.db")
        world = SocialMemoryWorld(store, _NoopModel())
        app = create_app(_settings(tmp_path), world)
        async with app.router.lifespan_context(app):
            async with _client(app) as client:
                _seed_space(store, "space1", "Space space1")
                store.overwrite_user_model(
                    "space1", {"summary": "loves coffee", "likes": ["espresso"]}
                )
                await _login(client)
                await scenario(store, client)

    asyncio.run(run())


def test_json_block_indents_escapes_and_passes_through_non_json():
    from gossipmemo.admin.render import json_block

    assert json_block('{"b":1,"a":2}') == (
        "<pre>{\n  &quot;a&quot;: 2,\n  &quot;b&quot;: 1\n}</pre>"
    )
    # Non-JSON is shown verbatim rather than swallowed.
    assert json_block("not json at all") == "<pre>not json at all</pre>"
    assert json_block(None) == "<pre></pre>"
    # Escaping still applies to values coming out of the database.
    assert "&lt;script&gt;" in json_block('{"x": "<script>alert(1)</script>"}')
    # Non-ASCII stays readable instead of turning into \\uXXXX escapes.
    assert "中文" in json_block('{"x": "\\u4e2d\\u6587"}')
