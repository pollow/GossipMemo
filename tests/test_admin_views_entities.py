from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
    configured = False

    async def aclose(self):
        return None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "world.db",
        llm_base_url="http://llm.test/v1",
        llm_api_key="key",
        llm_model="model",
        admin_password=ADMIN_PASSWORD,
    )


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _seed_space(store: SqliteWorldStore, space_id: str, name: str) -> dict:
    """A space with two people (one alias containing XSS), a relationship,
    a memory linking both, an open and a retired learning goal (one tagged
    to coverage root M1 via an entry_id, one untagged), a support+counter
    hypothesis, and coverage entries under M1 -- one root-level (path='')
    and one sub-path, so the root-vs-entry granularity rule is exercised.
    """

    store.ensure_space(space_id, name)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = now_iso()

    store.record_messages(
        space_id,
        [
            MessageInput(
                author="user",
                content="My friend Alice likes coffee",
                occurred_at=base,
                source=SourceRef(provider="test"),
            ),
        ],
    )

    memory_id = store.add_manual_memory(
        space_id,
        ManualMemoryRequest(
            content="Alice and Bob are close friends",
            kind="fact",
            people=["Alice", f"Bob {XSS}"],
            about_user=False,
        ),
    )

    with store._connect() as connection:
        person_rows = connection.execute(
            "SELECT p.id, p.display_name FROM people p "
            "JOIN memory_people mp ON mp.person_id = p.id WHERE mp.memory_id = ? "
            "ORDER BY p.display_name",
            (memory_id,),
        ).fetchall()
        people_by_name = {row["display_name"]: row["id"] for row in person_rows}
        alice_id = people_by_name["Alice"]
        bob_id = people_by_name[f"Bob {XSS}"]
        a_id, b_id = sorted((alice_id, bob_id))
        relationship_id = new_id("relationship")
        connection.execute(
            "INSERT INTO relationships(id, space_id, person_a_id, person_b_id, "
            "summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (relationship_id, space_id, a_id, b_id, "close friends", now, now),
        )
        connection.execute(
            "INSERT INTO memory_relationships(memory_id, relationship_id, role) "
            "VALUES (?, ?, 'about')",
            (memory_id, relationship_id),
        )

        # Coverage entries: one root overview (path='') and one sub-path,
        # both under root M1.
        root_entry_id = new_id("entry")
        connection.execute(
            "INSERT INTO coverage_entries(id, space_id, root, path, content, "
            "status, created_at, updated_at) VALUES (?, ?, 'M1', '', ?, 'active', ?, ?)",
            (root_entry_id, space_id, "Overview of life chapters", now, now),
        )
        sub_entry_id = new_id("entry")
        connection.execute(
            "INSERT INTO coverage_entries(id, space_id, root, path, content, "
            "status, created_at, updated_at) VALUES (?, ?, 'M1', 'childhood', ?, "
            "'active', ?, ?)",
            (sub_entry_id, space_id, "Grew up in the suburbs", now, now),
        )
        connection.execute(
            "UPDATE coverage_roots SET revision = 3, source_watermark = ? "
            "WHERE space_id = ? AND root = 'M1'",
            (now, space_id),
        )

        # Learning goals: one tagged to M1 via entry_ids, one untagged, with
        # an XSS-laden prompt on the tagged one to exercise escaping.
        goal_m1_id = new_id("goal")
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, "
            "focus_kind, status, created_at, updated_at) VALUES "
            "(?, ?, ?, 'because', ?, 'user', 'open', ?, ?)",
            (
                goal_m1_id,
                space_id,
                f"What was childhood like {XSS}?",
                f'["{sub_entry_id}"]',
                now,
                now,
            ),
        )
        goal_other_id = new_id("goal")
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, "
            "focus_kind, status, created_at, updated_at) VALUES "
            "(?, ?, 'What about work?', 'because', '[]', 'user', 'retired', ?, ?)",
            (goal_other_id, space_id, now, now),
        )

        # Hypotheses: one user-owned with a support memory and a counter
        # memory, one person-owned.
        counter_memory_id = new_id("memory")
        connection.execute(
            "INSERT INTO memories(id, space_id, content, kind, basis, about_user, "
            "status, created_by, created_at, updated_at) VALUES "
            "(?, ?, 'Actually Alice prefers tea', 'fact', 'explicit', 0, 'active', "
            "'llm', ?, ?)",
            (counter_memory_id, space_id, now, now),
        )
        hyp_user_id = new_id("hypothesis")
        connection.execute(
            "INSERT INTO hypotheses(id, space_id, owner_kind, content, kind, "
            "confidence, status, created_at, updated_at) VALUES "
            "(?, ?, 'user', 'User likes coffee', 'preference', 'medium', 'open', ?, ?)",
            (hyp_user_id, space_id, now, now),
        )
        connection.execute(
            "INSERT INTO hypothesis_evidence(hypothesis_id, memory_id, role) "
            "VALUES (?, ?, 'support')",
            (hyp_user_id, memory_id),
        )
        connection.execute(
            "INSERT INTO hypothesis_evidence(hypothesis_id, memory_id, role) "
            "VALUES (?, ?, 'counter')",
            (hyp_user_id, counter_memory_id),
        )
        hyp_person_id = new_id("hypothesis")
        connection.execute(
            "INSERT INTO hypotheses(id, space_id, owner_kind, owner_id, content, "
            "kind, confidence, status, created_at, updated_at) VALUES "
            "(?, ?, 'person', ?, 'Alice is outgoing', 'impression', 'low', 'open', "
            "?, ?)",
            (hyp_person_id, space_id, alice_id, now, now),
        )

    return {
        "memory_id": memory_id,
        "alice_id": alice_id,
        "bob_id": bob_id,
        "relationship_id": relationship_id,
        "goal_m1_id": goal_m1_id,
        "goal_other_id": goal_other_id,
        "hyp_user_id": hyp_user_id,
        "hyp_person_id": hyp_person_id,
        "root_entry_id": root_entry_id,
        "sub_entry_id": sub_entry_id,
    }


async def _login(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/admin/login", data={"password": ADMIN_PASSWORD}, follow_redirects=False
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


# --- people ---------------------------------------------------------------


def test_people_list_shows_aliases(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/people")
        assert response.status_code == 200
        assert "Alice" in response.text
        assert "&lt;script&gt;" in response.text
        assert "<script>" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_people_pagination_boundaries(tmp_path: Path):
    async def scenario(client, fixtures):
        first = await client.get("/admin/spaces/space1/people?offset=0&limit=1")
        assert first.status_code == 200
        assert "1-1 of 2" in first.text

        out_of_range = await client.get("/admin/spaces/space1/people?offset=1000&limit=1")
        assert out_of_range.status_code == 200
        assert "No results" in out_of_range.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_person_detail_shows_aliases_and_linked_memories(tmp_path: Path):
    async def scenario(client, fixtures):
        alice_id = fixtures["space1"]["alice_id"]
        response = await client.get(f"/admin/spaces/space1/people/{alice_id}")
        assert response.status_code == 200
        assert "Aliases" in response.text
        assert "Alice" in response.text
        assert "close friends" in response.text  # linked memory content

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_person_detail_unknown_id_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/people/does-not-exist")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- relationships ----------------------------------------------------------


def test_relationships_list_resolves_person_names(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/relationships")
        assert response.status_code == 200
        assert "Alice" in response.text
        assert "&lt;script&gt;" in response.text
        assert "<script>" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_relationship_detail_resolves_both_endpoints(tmp_path: Path):
    async def scenario(client, fixtures):
        relationship_id = fixtures["space1"]["relationship_id"]
        response = await client.get(f"/admin/spaces/space1/relationships/{relationship_id}")
        assert response.status_code == 200
        assert "Alice" in response.text
        assert "close friends" in response.text
        assert "Alice and Bob are close friends" in response.text  # linked memory

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_relationship_detail_unknown_id_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/relationships/does-not-exist")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_memory_detail_links_people_and_relationships(tmp_path: Path):
    async def scenario(client, fixtures):
        memory_id = fixtures["space1"]["memory_id"]
        alice_id = fixtures["space1"]["alice_id"]
        relationship_id = fixtures["space1"]["relationship_id"]
        response = await client.get(f"/admin/spaces/space1/memories/{memory_id}")
        assert response.status_code == 200
        assert f'/admin/spaces/space1/people/{alice_id}"' in response.text
        assert f'/admin/spaces/space1/relationships/{relationship_id}"' in response.text
        assert "not available yet" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- learning goals ---------------------------------------------------------


def test_goals_list_shows_prompt_and_status(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/goals")
        assert response.status_code == 200
        assert "What about work?" in response.text
        assert "&lt;script&gt;" in response.text
        assert "<script>" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_goals_filter_by_root_actually_filters(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/goals?root=M1")
        assert response.status_code == 200
        assert "1-1 of 1" in response.text
        assert "childhood" in response.text  # prompt text, escaped

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_goals_filter_by_status_actually_filters(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/goals?status=retired")
        assert response.status_code == 200
        assert "1-1 of 1" in response.text
        assert "What about work?" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_goals_unfiltered_shows_both(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/goals")
        assert "1-2 of 2" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- hypotheses --------------------------------------------------------------


def test_hypotheses_list_shows_support_and_counter_distinctly(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/hypotheses")
        assert response.status_code == 200
        assert "User likes coffee" in response.text
        assert "Alice and Bob are close friends" in response.text  # support evidence
        assert "Actually Alice prefers tea" in response.text  # counter evidence
        support_index = response.text.index("Support")
        counter_index = response.text.index("Counter")
        assert support_index < counter_index

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_hypotheses_filter_by_owner_kind(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/hypotheses?owner_kind=person")
        assert response.status_code == 200
        assert "1-1 of 1" in response.text
        assert "Alice is outgoing" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_hypotheses_unfiltered_shows_both(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/hypotheses")
        assert "1-2 of 2" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- coverage ----------------------------------------------------------------


def test_coverage_root_list_shows_all_roots_with_counts(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/coverage")
        assert response.status_code == 200
        assert "M1" in response.text
        assert "M9" in response.text
        assert "P11" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_coverage_drill_down_from_root_list_to_entries(tmp_path: Path):
    async def scenario(client, fixtures):
        roots = await client.get("/admin/spaces/space1/coverage")
        assert '/admin/spaces/space1/coverage/M1"' in roots.text

        entries = await client.get("/admin/spaces/space1/coverage/M1")
        assert entries.status_code == 200
        assert "1-2 of 2" in entries.text
        assert "Overview of life chapters" in entries.text
        assert "Grew up in the suburbs" in entries.text
        assert "(root overview)" in entries.text  # empty-path entry, same list

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_coverage_unknown_root_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/space1/coverage/not-a-root")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- cross-cutting: session + CSP on every new route -------------------------


@pytest.mark.parametrize(
    "path_template",
    [
        "/admin/spaces/space1/people",
        "/admin/spaces/space1/people/{alice_id}",
        "/admin/spaces/space1/relationships",
        "/admin/spaces/space1/relationships/{relationship_id}",
        "/admin/spaces/space1/goals",
        "/admin/spaces/space1/hypotheses",
        "/admin/spaces/space1/coverage",
        "/admin/spaces/space1/coverage/M1",
    ],
)
def test_every_new_view_requires_a_session(tmp_path: Path, path_template):
    async def scenario():
        store = SqliteWorldStore(tmp_path / "world.db")
        world = SocialMemoryWorld(store, _NoopModel())
        app = create_app(_settings(tmp_path), world)
        async with app.router.lifespan_context(app):
            fixture = _seed_space(store, "space1", "Space space1")
            path = path_template.format(**fixture)
            async with _client(app) as client:
                response = await client.get(path, follow_redirects=False)
                assert response.status_code == 303
                assert response.headers["location"] == "/admin/login"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "path_template",
    [
        "/admin/spaces/space1/people",
        "/admin/spaces/space1/people/{alice_id}",
        "/admin/spaces/space1/relationships",
        "/admin/spaces/space1/relationships/{relationship_id}",
        "/admin/spaces/space1/goals",
        "/admin/spaces/space1/hypotheses",
        "/admin/spaces/space1/coverage",
        "/admin/spaces/space1/coverage/M1",
    ],
)
def test_every_new_view_carries_csp_header(tmp_path: Path, path_template):
    async def scenario(client, fixtures):
        path = path_template.format(**fixtures["space1"])
        response = await client.get(path)
        assert response.status_code == 200
        assert response.headers.get("Content-Security-Policy")
        assert response.headers.get("X-Frame-Options") == "DENY"

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))
