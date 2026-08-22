from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from harness import XSS, SilentTransport, run_admin

from gossipmemo.transport import LLMRequestError


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


async def _run(
    tmp_path: Path,
    scenario,
    *,
    trace_dir: Path | None,
    authenticate: bool = True,
    playground_enabled: bool = False,
    transport=None,
):
    await run_admin(
        tmp_path,
        scenario,
        seed_spaces=[],
        settings_overrides={
            "llm_trace_path": trace_dir,
            "admin_playground_enabled": playground_enabled,
        },
        authenticate=authenticate,
        transport=transport,
    )


class FakePlaygroundTransport(SilentTransport):
    """A transport double for playground replay tests: answers or raises on demand.

    Subclasses `SilentTransport` for the protocol boilerplate (gate,
    context_budget, retry_policy, prepare); only `complete` is
    overridden, and it records every request it is asked to send so a
    test can assert on the edited content that actually went out.
    """

    def __init__(self, *, reply: str = "replayed completion", raise_error: bool = False):
        self.reply = reply
        self.raise_error = raise_error
        self.sent_requests: list = []

    async def complete(self, request, *, trace: bool = True):
        self.sent_requests.append(request)
        if self.raise_error:
            raise LLMRequestError("boom from provider")
        return self.reply


def _extract_csrf(text: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]*)"', text)
    assert match, "no csrf_token field found in rendered form"
    return match.group(1)


def _replay_form_fields(text: str, *, message_count: int) -> dict[str, str]:
    """Build a POST body from a rendered detail page's own prefilled values,
    used as the baseline that individual tests then edit."""

    fields = {
        "csrf_token": _extract_csrf(text),
        "message_count": str(message_count),
        "model": "test-model",
        "temperature": "",
        "max_tokens": "",
        "response_format": "",
    }
    for index in range(message_count):
        fields[f"message_{index}_role"] = "system" if index == 0 else "user"
        fields[f"message_{index}_content"] = "unused-placeholder"
    return fields


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


# --- Slice 3: the replay form, its gating, CSRF, and the compare view. ---


def test_replay_form_hidden_when_playground_disabled(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )

    async def scenario(client, fixtures):
        response = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        assert response.status_code == 200
        assert "GOSSIPMEMO_ADMIN_PLAYGROUND_ENABLED" in response.text
        assert "<form" not in response.text

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir, playground_enabled=False))


def test_replay_form_shown_when_playground_enabled(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )

    async def scenario(client, fixtures):
        response = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        assert response.status_code == 200
        assert f'action="/admin/playground/2026-08-21/{path.name}/run"' in response.text
        assert 'name="csrf_token"' in response.text
        # Prefilled from the historical record's messages.
        assert "You are a system." in response.text
        assert "hello" in response.text

    asyncio.run(
        _run(
            tmp_path, scenario, trace_dir=trace_dir, playground_enabled=True,
            transport=FakePlaygroundTransport(),
        )
    )


def test_post_run_404s_when_playground_disabled(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )

    async def scenario(client, fixtures):
        response = await client.post(
            f"/admin/playground/2026-08-21/{path.name}/run",
            data={"csrf_token": "anything"},
        )
        assert response.status_code == 404

    asyncio.run(_run(tmp_path, scenario, trace_dir=trace_dir, playground_enabled=False))


def test_post_run_rejected_without_valid_csrf(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )

    async def scenario(client, fixtures):
        detail = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        fields = _replay_form_fields(detail.text, message_count=2)
        fields["csrf_token"] = "not-the-real-token"
        response = await client.post(
            f"/admin/playground/2026-08-21/{path.name}/run", data=fields, follow_redirects=False,
        )
        assert response.status_code == 403

    asyncio.run(
        _run(
            tmp_path, scenario, trace_dir=trace_dir, playground_enabled=True,
            transport=FakePlaygroundTransport(),
        )
    )


def test_successful_replay_writes_playground_trace_not_production(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )
    transport = FakePlaygroundTransport(reply="replayed completion")

    async def scenario(client, fixtures):
        detail = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        fields = _replay_form_fields(detail.text, message_count=2)
        fields["message_1_content"] = "an edited user prompt"

        response = await client.post(
            f"/admin/playground/2026-08-21/{path.name}/run", data=fields, follow_redirects=False,
        )
        assert response.status_code == 303
        location = response.headers["location"]
        assert location.startswith(f"/admin/playground/2026-08-21/{path.name}/compare/")

        # The replay landed under the reserved playground/ subdirectory,
        # never a production day directory.
        playground_files = list((trace_dir / "playground").glob("*/*.json"))
        assert len(playground_files) == 1
        # `trace_dir.glob("*/*.json")` is exactly two segments deep, so it
        # only ever matches production day directories -- the reserved
        # `playground/<day>/<name>.json` tree is three segments deep and
        # never shows up here even without an explicit exclusion.
        production_files = list(trace_dir.glob("*/*.json"))
        assert len(production_files) == 1  # only the original seed trace

        record = json.loads(playground_files[0].read_text(encoding="utf-8"))
        assert record["completion"] == "replayed completion"
        assert record["request"]["messages"][1]["content"] == "an edited user prompt"
        assert record["error"] is None

        assert len(transport.sent_requests) == 1
        assert transport.sent_requests[0].messages[1].content == "an edited user prompt"

        compare = await client.get(location)
        assert compare.status_code == 200
        assert "hello" in compare.text  # original's message content
        assert "an edited user prompt" in compare.text  # replay's edited content
        assert "replayed completion" in compare.text  # replay's completion
        assert "<pre>ok</pre>" in compare.text  # original's completion, from the seed fixture

    asyncio.run(
        _run(tmp_path, scenario, trace_dir=trace_dir, playground_enabled=True, transport=transport)
    )


def test_llm_failure_renders_error_not_500(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )
    transport = FakePlaygroundTransport(raise_error=True)

    async def scenario(client, fixtures):
        detail = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        fields = _replay_form_fields(detail.text, message_count=2)

        response = await client.post(
            f"/admin/playground/2026-08-21/{path.name}/run", data=fields, follow_redirects=False,
        )
        assert response.status_code == 303
        compare = await client.get(response.headers["location"])
        assert compare.status_code == 200
        assert "boom from provider" in compare.text

    asyncio.run(
        _run(tmp_path, scenario, trace_dir=trace_dir, playground_enabled=True, transport=transport)
    )


@pytest.mark.parametrize(
    "day",
    ["..%2f..", "2026-08-21%2f..", "not-a-day"],
)
def test_path_traversal_in_day_is_rejected_on_run_route(tmp_path: Path, day):
    trace_dir = tmp_path / "traces"
    _write_trace(trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve this", encoding="utf-8")

    async def scenario(client, fixtures):
        response = await client.post(
            f"/admin/playground/{day}/name/run", data={"csrf_token": "x"},
        )
        assert response.status_code == 404
        assert "do not serve this" not in response.text

    asyncio.run(
        _run(
            tmp_path, scenario, trace_dir=trace_dir, playground_enabled=True,
            transport=FakePlaygroundTransport(),
        )
    )


def test_replay_history_browsing_works(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    path = _write_trace(
        trace_dir / "2026-08-21", time_prefix="090000000", label="extract", sequence=1,
    )
    transport = FakePlaygroundTransport(reply="a replay completion")

    async def scenario(client, fixtures):
        # No replays yet: the replay day list shows the friendly empty state,
        # not a 500.
        empty = await client.get("/admin/playground/replays")
        assert empty.status_code == 200

        detail = await client.get(f"/admin/playground/2026-08-21/{path.name}")
        fields = _replay_form_fields(detail.text, message_count=2)
        run_response = await client.post(
            f"/admin/playground/2026-08-21/{path.name}/run", data=fields, follow_redirects=False,
        )
        assert run_response.status_code == 303

        days_response = await client.get("/admin/playground/replays")
        assert days_response.status_code == 200
        match = re.search(r'href="(/admin/playground/replays/[\d-]+)"', days_response.text)
        assert match, days_response.text
        day_url = match.group(1)

        day_response = await client.get(day_url)
        assert day_response.status_code == 200
        call_match = re.search(r'href="(' + re.escape(day_url) + r'/[^"]+)"', day_response.text)
        assert call_match, day_response.text

        call_response = await client.get(call_match.group(1))
        assert call_response.status_code == 200
        assert "a replay completion" in call_response.text

    asyncio.run(
        _run(tmp_path, scenario, trace_dir=trace_dir, playground_enabled=True, transport=transport)
    )
