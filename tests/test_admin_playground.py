from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from harness import XSS, run_admin


def _write_trace(day_dir: Path, *, time_prefix: str, label: str, sequence: int,
                 status: int = 200, error: str | None = None, completion: str | None = "ok",
                 content: str = "hello") -> Path:
    day_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "sequence": sequence,
        "timestamp": "2026-08-21T12:00:00+00:00",
        "label": label,
        "tier": 1,
        "model": "test-model",
        "status": status,
        "estimated_tokens": 42,
        "request": {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a system."},
                {"role": "user", "content": content},
            ],
        },
        "completion": completion,
        "error": error,
    }
    name = f"{time_prefix}-{label}-{sequence:04d}.json"
    path = day_dir / name
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


async def _run(tmp_path: Path, scenario, *, trace_dir: Path | None, authenticate: bool = True):
    await run_admin(
        tmp_path,
        scenario,
        seed_spaces=[],
        settings_overrides={"llm_trace_path": trace_dir},
        authenticate=authenticate,
    )


def test_unset_trace_path_shows_friendly_message(tmp_path: Path):
    async def scenario(client, fixtures):
        response = await client.get("/admin/playground")
        assert response.status_code == 200
        assert "GOSSIPMEMO_LLM_TRACE_PATH" in response.text
        assert "<table" not in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=None))


def test_missing_trace_directory_shows_friendly_message(tmp_path: Path):
    trace_dir = tmp_path / "does-not-exist"

    async def scenario(client, fixtures):
        response = await client.get("/admin/playground")
        assert response.status_code == 200
        assert "does not exist" in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


def test_day_listing_newest_first_and_excludes_reserved_playground_dir(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    _write_trace(trace_dir / "2026-08-20", time_prefix="090000000", label="extract", sequence=1)
    _write_trace(trace_dir / "2026-08-21", time_prefix="100000000", label="extract", sequence=1)
    _write_trace(trace_dir / "2026-08-21", time_prefix="110000000", label="continuity", sequence=2)
    # The reserved subdirectory a later slice writes replay traces under.
    # It must never appear as a "day".
    (trace_dir / "playground").mkdir()
    (trace_dir / "playground" / "not-a-day.json").write_text("{}", encoding="utf-8")

    async def scenario(client, fixtures):
        response = await client.get("/admin/playground")
        assert response.status_code == 200
        text = response.text
        assert "2026-08-21" in text
        assert "2026-08-20" in text
        assert text.index("2026-08-21") < text.index("2026-08-20")
        assert '>playground<' not in text
        assert 'href="/admin/playground/playground"' not in text
        # 2026-08-21 had two calls.
        assert ">2<" in text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


def test_day_with_no_files_is_empty_but_not_broken(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    (trace_dir / "2026-08-21").mkdir(parents=True)

    async def scenario(client, fixtures):
        response = await client.get("/admin/playground/2026-08-21")
        assert response.status_code == 200
        assert "No calls traced" in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


def test_call_listing_shows_newest_first_with_columns(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
        status=200,
    )
    _write_trace(
        trace_dir / "2026-08-21", time_prefix="110000000", label="continuity", sequence=2,
        status=500, error="boom",
    )

    async def scenario(client, fixtures):
        response = await client.get("/admin/playground/2026-08-21")
        assert response.status_code == 200
        text = response.text
        assert "continuity" in text
        assert "extract" in text
        assert text.index("continuity") < text.index("extract")
        assert "test-model" in text
        assert "boom" not in text  # error text itself isn't on the list page
        assert ">yes<" in text  # continuity row has an error
        assert ">no<" in text  # extract row does not

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


def test_detail_renders_messages_and_completion_with_escaping(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
        content=f"Tell me about {XSS}", completion="The completion text",
    )

    async def scenario(client, fixtures):
        response = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        assert response.status_code == 200
        text = response.text
        assert "You are a system." in text
        assert "The completion text" in text
        assert "<script>" not in text
        assert "&lt;script&gt;" in text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


def test_detail_shows_error_when_present(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
        status=500, error="upstream exploded", completion=None,
    )

    async def scenario(client, fixtures):
        response = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        assert response.status_code == 200
        assert "upstream exploded" in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


def test_malformed_json_file_is_skipped_not_a_500(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    day_dir = trace_dir / "2026-08-21"
    day_dir.mkdir(parents=True)
    good = _write_trace(day_dir, time_prefix="090000000", label="extract", sequence=1)
    bad = day_dir / "080000000-extract-0002.json"
    bad.write_text("{not valid json", encoding="utf-8")

    async def scenario(client, fixtures):
        list_response = await client.get("/admin/playground/2026-08-21")
        assert list_response.status_code == 200  # not a 500

        detail_response = await client.get(f"/admin/playground/2026-08-21/{good.name}")
        assert detail_response.status_code == 200

        bad_detail = await client.get(f"/admin/playground/2026-08-21/{bad.name}")
        assert bad_detail.status_code == 200  # shown as an error, not raised
        assert bad_detail.status_code != 500

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


@pytest.mark.parametrize(
    "day",
    [
        "..%2f..",
        "2026-08-21%2f..",
        "not-a-day",
        "2026-08-21x",
        "....",
    ],
)
def test_path_traversal_in_day_is_rejected(tmp_path: Path, day):
    trace_dir = tmp_path / "traces"
    _write_trace(trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve this", encoding="utf-8")

    async def scenario(client, fixtures):
        response = await client.get(f"/admin/playground/{day}")
        assert response.status_code == 404
        assert "do not serve this" not in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


@pytest.mark.parametrize(
    "name",
    [
        "..%2f..%2fsecret.txt",
        "not-a-real-file.json",
        "090000000-extract-0001.json%2f..%2f..%2fsecret.txt",
        "secret.txt",
    ],
)
def test_path_traversal_in_name_is_rejected(tmp_path: Path, name):
    trace_dir = tmp_path / "traces"
    _write_trace(trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve this", encoding="utf-8")

    async def scenario(client, fixtures):
        response = await client.get(f"/admin/playground/2026-08-21/{name}")
        assert response.status_code == 404
        assert "do not serve this" not in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir))


@pytest.mark.parametrize(
    "path",
    [
        "/admin/playground",
        "/admin/playground/2026-08-21",
        "/admin/playground/2026-08-21/090000000-extract-0001.json",
    ],
)
def test_every_playground_view_requires_a_session(tmp_path: Path, path):
    trace_dir = tmp_path / "traces"
    _write_trace(trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1)

    async def scenario(client, fixtures):
        response = await client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir, authenticate=False))
