from __future__ import annotations

import asyncio
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest
from harness import XSS, run_admin

from gossipmemo.models import MessageInput, SourceRef
from gossipmemo.store import SqliteWorldStore
from gossipmemo.store.policy import new_id, now_iso


async def _run(tmp_path: Path, scenario, *, seed_spaces: list[str] | None = None):
    await run_admin(tmp_path, scenario, seeder=_seed_space, seed_spaces=seed_spaces or [])


def _seed_space(store: SqliteWorldStore, space_id: str, name: str) -> dict:
    """A space with one message, one person, a continuity snapshot pointing
    at that message and person (both carrying XSS to exercise escaping), one
    extraction batch, and one embedding row with a known unit vector so its
    L2 norm is a predictable `1.000000`."""

    store.ensure_space(space_id, name)
    now = now_iso()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    store.record_messages(
        space_id,
        [
            MessageInput(
                author="user",
                content="Message content about Alice birthday plans",
                occurred_at=base,
                source=SourceRef(provider="test"),
            ),
        ],
    )

    with store._connect() as connection:
        message_id = connection.execute(
            "SELECT id FROM messages WHERE space_id = ? ORDER BY rowid LIMIT 1", (space_id,)
        ).fetchone()["id"]

        person_id = new_id("person")
        connection.execute(
            "INSERT INTO people(id, space_id, display_name, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'active', ?, ?)",
            (person_id, space_id, f"Alice {XSS}", now, now),
        )

        connection.execute(
            "UPDATE continuities SET text = ?, related_person_ids = ?, "
            "through_message_id = ?, updated_at = ? WHERE space_id = ?",
            (
                f"Ongoing thread about Alice birthday {XSS}",
                f'["{person_id}"]',
                message_id,
                now,
                space_id,
            ),
        )

        batch_id = new_id("batch")
        connection.execute(
            "INSERT INTO extraction_batches(id, space_id, created_at, completed_at) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, space_id, now, now),
        )

        # A unit vector -- (0.6, 0.8, 0.0) -- so the rendered L2 norm is
        # predictably "1.000000" and the raw components (0.6 / 0.8) are
        # easy to assert absent from the response body.
        vector = struct.pack("<3f", 0.6, 0.8, 0.0)
        embedding_owner_id = new_id("memory")
        connection.execute(
            "INSERT INTO embeddings(space_id, owner_kind, owner_id, model, dim, vector, "
            "content_hash, created_at) VALUES (?, 'memory', ?, 'test-model', 3, ?, 'hash', ?)",
            (space_id, embedding_owner_id, vector, now),
        )

    return {
        "message_id": message_id,
        "person_id": person_id,
        "batch_id": batch_id,
        "embedding_owner_id": embedding_owner_id,
    }


# --- continuity on the space overview ----------------------------------------


def test_space_overview_renders_continuity_text_message_and_linked_people(tmp_path: Path):
    """`continuities` keys on `space_id`, so a space holds exactly one row
    that reasoning overwrites in place. There is no history to page through,
    and the overview is the only place continuity is shown."""

    async def scenario(client, fixtures):
        person_id = fixtures["space1"]["person_id"]
        response = await client.get("/admin/spaces/space1")
        assert response.status_code == 200
        assert "Ongoing thread about Alice birthday" in response.text
        assert "Message content about Alice birthday plans" in response.text
        assert f'/admin/spaces/space1/people/{person_id}"' in response.text
        assert "<script>" not in response.text
        assert "&lt;script&gt;" in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- raw tables --------------------------------------------------------------


def test_admin_tables_index_lists_the_three_whitelisted_tables(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/tables")
        assert response.status_code == 200
        for name in ("schema_migrations", "extraction_batches", "embeddings"):
            assert name in response.text
            assert f'href="/admin/tables/{name}"' in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_index_linked_from_space_list(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/spaces")
        assert response.status_code == 200
        assert 'href="/admin/tables"' in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_schema_migrations_renders_rows(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/tables/schema_migrations")
        assert response.status_code == 200
        assert "No results." not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_extraction_batches_renders_rows(tmp_path: Path):
    async def scenario(client, fixtures):
        batch_id = fixtures["space1"]["batch_id"]
        response = await client.get("/admin/tables/extraction_batches")
        assert response.status_code == 200
        assert batch_id in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_embeddings_shows_dim_and_norm_not_raw_floats(tmp_path: Path):
    async def scenario(client, fixtures):
        owner_id = fixtures["space1"]["embedding_owner_id"]
        response = await client.get("/admin/tables/embeddings")
        assert response.status_code == 200
        assert owner_id in response.text
        assert "<td>3</td>" in response.text  # dim
        assert "1.000000" in response.text  # L2 norm of (0.6, 0.8, 0.0)
        assert "0.6" not in response.text
        assert "0.8" not in response.text

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_unknown_name_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/tables/does-not-exist")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_non_whitelisted_real_table_returns_404(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/tables/memories")
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


def test_admin_tables_sql_injection_shaped_name_returns_404_and_does_not_execute(
    tmp_path: Path,
):
    async def scenario(client, fixtures):
        response = await client.get("/admin/tables/memories; DROP TABLE memories")
        assert response.status_code == 404

        # The database must be untouched: an ordinary page still works.
        sane = await client.get("/admin/spaces/space1")
        assert sane.status_code == 200

    asyncio.run(_run(tmp_path, scenario, seed_spaces=["space1"]))


# --- cross-cutting: session + CSP on every new route -------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/admin/spaces/space1",
        "/admin/tables",
        "/admin/tables/schema_migrations",
        "/admin/tables/extraction_batches",
        "/admin/tables/embeddings",
    ],
)
def test_every_slice5_view_requires_a_session(tmp_path: Path, path):
    async def scenario(client, fixtures):
        response = await client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"

    asyncio.run(run_admin(tmp_path, scenario, seeder=_seed_space,
                          seed_spaces=["space1"], authenticate=False))
