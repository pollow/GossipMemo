from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from harness import XSS, run_admin

from gossipmemo.models import MessageInput, SourceRef
from gossipmemo.store import SqliteWorldStore
from gossipmemo.store.policy import new_id, now_iso


def _insert_memory(connection, space_id: str, content: str, now: str) -> str:
    memory_id = new_id("memory")
    connection.execute(
        "INSERT INTO memories(id, space_id, content, kind, basis, about_user, "
        "status, created_by, created_at, updated_at) VALUES "
        "(?, ?, ?, 'fact', 'explicit', 0, 'active', 'llm', ?, ?)",
        (memory_id, space_id, content, now, now),
    )
    return memory_id


def _seed_space(store: SqliteWorldStore, space_id: str, name: str) -> dict:
    """A space carrying the shared keyword "sunrise" across all seven
    searchable kinds, plus dedicated fixtures for the LIKE-metacharacter,
    case, Chinese, and alias-dedup edge cases.
    """

    store.ensure_space(space_id, name)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = now_iso()

    memory_id = None
    xss_memory_id = None
    percent_memory_id = None
    control_100_memory_id = None
    ab_memory_id = None
    axb_memory_id = None
    backslash_memory_id = None
    chinese_memory_id = None

    with store._connect() as connection:
        memory_id = _insert_memory(connection, space_id, "Watched the sunrise together", now)
        xss_memory_id = _insert_memory(connection, space_id, f"Payload {XSS} arrived", now)
        percent_memory_id = _insert_memory(connection, space_id, "Progress is 100% done", now)
        control_100_memory_id = _insert_memory(connection, space_id, "There were 100 apples", now)
        ab_memory_id = _insert_memory(connection, space_id, "The variable a_b is set", now)
        axb_memory_id = _insert_memory(connection, space_id, "The value axb appears here", now)
        backslash_memory_id = _insert_memory(
            connection, space_id, "Path is C:\\temp\\file for backups", now
        )
        chinese_memory_id = _insert_memory(connection, space_id, "你好世界，这是一个测试", now)

        # Person matched only by alias.
        riley_id = new_id("person")
        connection.execute(
            "INSERT INTO people(id, space_id, display_name, status, created_at, updated_at) "
            "VALUES (?, ?, 'Riley', 'active', ?, ?)",
            (riley_id, space_id, now, now),
        )
        connection.execute(
            "INSERT INTO person_aliases(id, space_id, person_id, value, normalized_value) "
            "VALUES (?, ?, ?, 'Sunrise Runner', 'sunrise runner')",
            (new_id("alias"), space_id, riley_id),
        )

        # Person matched on both name and alias -- must still appear once.
        sunrise_bob_id = new_id("person")
        connection.execute(
            "INSERT INTO people(id, space_id, display_name, status, created_at, updated_at) "
            "VALUES (?, ?, 'Sunrise Bob', 'active', ?, ?)",
            (sunrise_bob_id, space_id, now, now),
        )
        connection.execute(
            "INSERT INTO person_aliases(id, space_id, person_id, value, normalized_value) "
            "VALUES (?, ?, ?, 'Sunrise Extra', 'sunrise extra')",
            (new_id("alias"), space_id, sunrise_bob_id),
        )

        goal_id = new_id("goal")
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, "
            "focus_kind, status, created_at, updated_at) VALUES "
            "(?, ?, 'What time is sunrise for you?', 'because', '[]', 'user', 'open', ?, ?)",
            (goal_id, space_id, now, now),
        )

        hyp_id = new_id("hypothesis")
        connection.execute(
            "INSERT INTO hypotheses(id, space_id, owner_kind, content, kind, "
            "confidence, status, created_at, updated_at) VALUES "
            "(?, ?, 'user', 'User enjoys sunrise walks', 'preference', 'medium', 'open', ?, ?)",
            (hyp_id, space_id, now, now),
        )

        entry_id = new_id("entry")
        connection.execute(
            "INSERT INTO coverage_entries(id, space_id, root, path, content, status, "
            "created_at, updated_at) VALUES "
            "(?, ?, 'M1', 'mornings', 'They mentioned a sunrise ritual', 'active', ?, ?)",
            (entry_id, space_id, now, now),
        )

        connection.execute(
            "UPDATE continuities SET text = 'Ongoing thread about the sunrise walk plan', "
            "updated_at = ? WHERE space_id = ?",
            (now, space_id),
        )

    message_ids = store.record_messages(
        space_id,
        [
            MessageInput(
                author="user",
                content="I love a sunrise walk",
                occurred_at=base,
                source=SourceRef(provider="test"),
            )
        ],
    )

    return {
        "memory_id": memory_id,
        "xss_memory_id": xss_memory_id,
        "percent_memory_id": percent_memory_id,
        "control_100_memory_id": control_100_memory_id,
        "ab_memory_id": ab_memory_id,
        "axb_memory_id": axb_memory_id,
        "backslash_memory_id": backslash_memory_id,
        "chinese_memory_id": chinese_memory_id,
        "riley_id": riley_id,
        "sunrise_bob_id": sunrise_bob_id,
        "goal_id": goal_id,
        "hyp_id": hyp_id,
        "entry_id": entry_id,
        "message_id": message_ids[0],
    }


def _seed_cap_space(store: SqliteWorldStore, space_id: str, name: str) -> None:
    store.ensure_space(space_id, name)
    now = now_iso()
    with store._connect() as connection:
        for index in range(25):
            _insert_memory(connection, space_id, f"Memory number {index} with capkw123", now)


async def _run(tmp_path: Path, scenario, *, seed_spaces=None, seed_cap: bool = False):
    await run_admin(
        tmp_path, scenario, seeder=_seed_space, seed_spaces=seed_spaces or [],
        also_seed=(lambda store: _seed_cap_space(store, "capspace", "Cap Space"))
        if seed_cap else None,
    )


def test_keyword_matches_across_kinds_grouped(tmp_path: Path):
    async def scenario(client, fixtures):
        f = fixtures["space1"]
        response = await client.get("/admin/spaces/space1/search?q=sunrise")
        assert response.status_code == 200
        body = response.text
        assert "Memories (1)" in body
        assert "Messages (1)" in body
        # Two distinct people: alias-only Riley and name+alias Sunrise Bob.
        assert "People (2)" in body
        assert "Learning goals (1)" in body
        assert "Hypotheses (1)" in body
        assert "Coverage entries (1)" in body
        assert "Continuity (1)" in body
        assert f"/admin/spaces/space1/memories/{f['memory_id']}" in body
        assert f"/admin/spaces/space1/people/{f['riley_id']}" in body
        assert f"/admin/spaces/space1/people/{f['sunrise_bob_id']}" in body

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_empty_query_shows_prompt_not_full_dump(tmp_path: Path):
    async def scenario(client, fixtures):
        for q in ("", "   "):
            response = await client.get(f"/admin/spaces/space1/search?q={q}")
            assert response.status_code == 200
            assert "Enter a keyword" in response.text
            assert "Watched the sunrise together" not in response.text
            assert "Memories (" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


@pytest.mark.parametrize(
    "query,hit,miss",
    [
        # LIKE metacharacters must be escaped, not act as wildcards -- each
        # case carries a control record that must NOT match.
        ("100%25", "percent_memory_id", "control_100_memory_id"),
        ("a_b", "ab_memory_id", "axb_memory_id"),
        # Backslash is a literal keyword, not a LIKE escape character.
        ("C%3A%5Ctemp", "backslash_memory_id", None),
        ("SUNRISE", "memory_id", None),  # ASCII case-insensitive
        ("%E4%BD%A0%E5%A5%BD", "chinese_memory_id", None),  # CJK matches literally
    ],
)
def test_keyword_matching_is_literal(tmp_path: Path, query, hit, miss):
    async def scenario(client, fixtures):
        f = fixtures["space1"]
        response = await client.get(f"/admin/spaces/space1/search?q={query}")
        assert response.status_code == 200
        assert f"/admin/spaces/space1/memories/{f[hit]}" in response.text
        if miss is not None:
            assert f"/admin/spaces/space1/memories/{f[miss]}" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_a_person_matched_more_than_once_is_listed_once(tmp_path: Path):
    """Riley matches by alias only; Bob matches by both name and alias."""

    async def scenario(client, fixtures):
        f = fixtures["space1"]
        response = await client.get("/admin/spaces/space1/search?q=sunrise")
        assert response.status_code == 200
        for key in ("riley_id", "sunrise_bob_id"):
            assert response.text.count(f"/admin/spaces/space1/people/{f[key]}") == 1

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_cap_holds_and_refine_note_appears(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/capspace/search?q=capkw123")
        assert response.status_code == 200
        body = response.text
        assert "Memories (20)" in body
        assert "refine your keyword" in body

    asyncio.run(_run(tmp_path, scenario, seed_cap=True))


def test_result_links_resolve_to_real_pages(tmp_path: Path):
    async def scenario(client, fixtures):
        f = fixtures["space1"]
        response = await client.get("/admin/spaces/space1/search?q=sunrise")
        assert response.status_code == 200

        memory_href = f"/admin/spaces/space1/memories/{f['memory_id']}"
        person_href = f"/admin/spaces/space1/people/{f['riley_id']}"
        assert memory_href in response.text
        assert person_href in response.text

        memory_response = await client.get(memory_href)
        assert memory_response.status_code == 200
        person_response = await client.get(person_href)
        assert person_response.status_code == 200

        for path in (
            "/admin/spaces/space1/messages",
            "/admin/spaces/space1/goals",
            "/admin/spaces/space1/hypotheses",
            "/admin/spaces/space1/coverage/M1",
            "/admin/spaces/space1",
        ):
            follow = await client.get(path)
            assert follow.status_code == 200

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_escaping_content_is_rendered_safely(tmp_path: Path):
    async def scenario(client, fixtures):
        f = fixtures["space1"]
        response = await client.get("/admin/spaces/space1/search?q=Payload")
        assert response.status_code == 200
        body = response.text
        assert f"/admin/spaces/space1/memories/{f['xss_memory_id']}" in body
        assert "&lt;script&gt;" in body
        assert "<script>alert(1)</script>" not in body

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_search_requires_a_session(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get(
            "/admin/spaces/space1/search?q=sunrise", follow_redirects=False
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"

    asyncio.run(run_admin(tmp_path, scenario, seeder=_seed_space,
                          seed_spaces=["space1"], authenticate=False))


def test_search_unknown_space_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces/does-not-exist/search?q=sunrise")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))
