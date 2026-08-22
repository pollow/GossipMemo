"""Admin views: the reasoner playground.

Slice 2 (browsing) renders the per-call JSON trace files
`OpenAICompatibleAdapter._trace` writes under `Settings.llm_trace_path`
(see `src/gossipmemo/llm.py`):

    <trace_dir>/<YYYY-MM-DD>/<HHMMSSmmm>-<label>-<sequence>.json

Slice 3 (this module, current state) adds an editable replay form on the
detail page. It is gated separately from browsing by
`Settings.admin_playground_enabled` (env `GOSSIPMEMO_ADMIN_PLAYGROUND_ENABLED`,
default off): when disabled, the detail page renders without the form and
the POST route 404s. Browsing itself (the routes ported from slice 2, plus
the mirrored replay-history routes below) is never gated -- it makes no LLM
call and writes nothing.

A replay does not reuse the trace `OpenAICompatibleAdapter._trace` writes
for production calls: it calls the shared adapter with `complete(...,
trace=False)` (so the call still serializes behind the one `ProviderGate`
that also arbitrates background reasoning -- see `reasoners/base.py`), then
writes its own record, in the same shape, via `llm.build_trace_record` and
`llm.write_trace_file`, into the reserved sibling directory
`_DAY_RE` below deliberately does not match:

    <trace_dir>/playground/<YYYY-MM-DD>/<HHMMSSmmm>-<label>-<sequence>.json

`day` and `name` (and the replay-history `day`/`name` pair) arrive from the
URL and would otherwise be joined onto a filesystem path taken from
settings. Both are validated against a strict regex *before* any `Path`
join, and the resulting path is additionally resolved and checked to still
be inside its root, as a second line of defense against traversal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from itertools import count
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from ...llm import build_trace_record, write_trace_file
from ...priority import TIER_FOREGROUND, llm_call_tier
from ...transport import ChatCompletionRequest, ChatMessage, LLMError, LlmTransport
from ..render import esc, html_response, page, table_component
from ._common import clamp_limit, clamp_offset

#: Day directories only. Excludes the reserved `playground/` subdirectory
#: replay traces live under, and anything else that isn't a day.
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: `<HHMMSSmmm>-<label>-<sequence>[-<attempt>].json`, matching exactly what
#: `llm.write_trace_file` (and, before slice 3, `_trace`) writes: a 9-digit
#: local-time prefix, a label reduced to `[a-z0-9-]`, a 4-digit sequence,
#: and an optional numeric collision suffix.
_NAME_RE = re.compile(r"^\d{9}-[a-z0-9-]+-\d{4}(-\d+)?\.json$")

_BREADCRUMB_ROOT = [("Admin", "/admin"), ("Playground", "/admin/playground")]
_REPLAY_BREADCRUMB_ROOT = _BREADCRUMB_ROOT + [("Replays", "/admin/playground/replays")]

#: Sequence numbers for replay trace files, independent of any adapter's own
#: `_trace_sequence` -- a replay is never written by `OpenAICompatibleAdapter`
#: itself, so it needs its own per-process counter.
_playground_sequence = count(1)


def register(
    router: APIRouter,
    require_session,
    trace_dir: Path | None,
    *,
    enabled: bool = False,
    model: LlmTransport | None = None,
    csrf_token: Callable[[], str] | None = None,
    verify_csrf: Callable[[str | None], bool] | None = None,
) -> None:
    """Attach the playground routes to `router`.

    `trace_dir` is `Settings.llm_trace_path`, passed in rather than
    re-read from settings so this module never imports `Settings`.
    `enabled` is `Settings.admin_playground_enabled`; `model` is the same
    `LlmTransport` (in production, the one `OpenAICompatibleAdapter`) the
    rest of the app calls, so a replay serializes behind the one
    `ProviderGate`. `csrf_token`/`verify_csrf` are `AdminAuth.csrf_token`/
    `AdminAuth.verify_csrf`, passed in so this module never constructs its
    own `AdminAuth`.
    """

    def _playground_root() -> Path | None:
        return trace_dir / "playground" if trace_dir is not None else None

    # --- Replay-history browsing (mirrors the day/call-list/detail routes
    # below, over the reserved `playground/` subdirectory instead of
    # `trace_dir` itself). Registered before the generic `/playground/{day}`
    # routes so the literal `replays` segment is not swallowed as a `day`.

    @router.get("/playground/replays", include_in_schema=False)
    async def playground_replay_days(_: None = Depends(require_session)) -> HTMLResponse:
        state, days = _list_days(_playground_root())
        if state != "ok":
            return html_response(
                page(
                    title="Playground replays",
                    breadcrumbs=_REPLAY_BREADCRUMB_ROOT,
                    body=_empty_state_body(state),
                )
            )
        rows = [[day, count_] for day, count_ in days]
        table_html = table_component(
            headers=["Day", "Calls"],
            rows=rows,
            column_classes=["nowrap mono", "num nowrap"],
            row_hrefs=[f"/admin/playground/replays/{day}" for day, _count in days],
            offset=0,
            limit=max(len(rows), 1),
            total=len(rows),
            base_path="/admin/playground/replays",
        )
        return html_response(
            page(title="Playground replays", breadcrumbs=_REPLAY_BREADCRUMB_ROOT, body=table_html)
        )

    @router.get("/playground/replays/{day}", include_in_schema=False)
    async def playground_replay_day(
        day: str, request: Request, _: None = Depends(require_session),
    ) -> HTMLResponse:
        if not _DAY_RE.match(day):
            return _not_found(_REPLAY_BREADCRUMB_ROOT, "No such day.")
        state, day_path = _resolve_day(_playground_root(), day)
        breadcrumbs = _REPLAY_BREADCRUMB_ROOT + [(day, f"/admin/playground/replays/{day}")]
        if state != "ok":
            return html_response(
                page(
                    title=f"Replays: {day}", breadcrumbs=breadcrumbs,
                    body=_empty_state_body(state),
                )
            )
        records = _load_day_records(day_path)
        if not records:
            return html_response(
                page(
                    title=f"Replays: {day}", breadcrumbs=breadcrumbs,
                    body=_empty_state_body("empty"),
                )
            )
        query = request.query_params
        clamped_offset = clamp_offset(query.get("offset"))
        clamped_limit = clamp_limit(query.get("limit"))
        page_records = records[clamped_offset: clamped_offset + clamped_limit]
        rows = [
            [
                entry["local_time"], entry.get("label", "-"), entry.get("model", "-"),
                entry.get("tier", "-"), entry.get("status", "-"),
                entry.get("estimated_tokens", "-"), "yes" if entry.get("error") else "no",
            ]
            for entry in page_records
        ]
        row_hrefs = [f"/admin/playground/replays/{day}/{entry['name']}" for entry in page_records]
        base_path = f"/admin/playground/replays/{day}"
        table_html = table_component(
            headers=["Local time", "Label", "Model", "Tier", "Status", "Est. tokens", "Error?"],
            rows=rows,
            column_classes=["nowrap mono", "nowrap", "nowrap", "num nowrap", "num nowrap",
                            "num nowrap", "nowrap"],
            row_hrefs=row_hrefs,
            offset=clamped_offset,
            limit=clamped_limit,
            total=len(records),
            base_path=base_path,
        )
        return html_response(
            page(title=f"Replays: {day}", breadcrumbs=breadcrumbs, body=table_html)
        )

    @router.get("/playground/replays/{day}/{name}", include_in_schema=False)
    async def playground_replay_call(
        day: str, name: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        if not _DAY_RE.match(day) or not _NAME_RE.match(name):
            return _not_found(_REPLAY_BREADCRUMB_ROOT, "No such call.")
        state, day_path = _resolve_day(_playground_root(), day)
        if state != "ok":
            return _not_found(_REPLAY_BREADCRUMB_ROOT, "No such call.")
        target = _safe_join(day_path, name)
        if target is None or not target.is_file():
            return _not_found(_REPLAY_BREADCRUMB_ROOT, "No such call.")
        breadcrumbs = _REPLAY_BREADCRUMB_ROOT + [
            (day, f"/admin/playground/replays/{day}"),
            (name, f"/admin/playground/replays/{day}/{name}"),
        ]
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return html_response(
                page(
                    title=f"Replay call: {name}", breadcrumbs=breadcrumbs,
                    body="<p>This trace file could not be parsed as JSON.</p>",
                )
            )
        body = _render_call_detail(record, include_replay_section=False)
        return html_response(page(title=f"Replay call: {name}", breadcrumbs=breadcrumbs, body=body))

    # --- Production trace browsing (slice 2), plus the slice-3 replay form
    # and compare view nested under a call's detail page.

    @router.get("/playground", include_in_schema=False)
    async def playground_days(_: None = Depends(require_session)) -> HTMLResponse:
        state, days = _list_days(trace_dir)
        if state != "ok":
            return html_response(
                page(
                    title="Reasoner playground",
                    breadcrumbs=_BREADCRUMB_ROOT,
                    body=_empty_state_body(state),
                )
            )
        rows = [[day, count_] for day, count_ in days]
        table_html = table_component(
            headers=["Day", "Calls"],
            rows=rows,
            column_classes=["nowrap mono", "num nowrap"],
            row_hrefs=[f"/admin/playground/{day}" for day, _count in days],
            offset=0,
            limit=max(len(rows), 1),
            total=len(rows),
            base_path="/admin/playground",
        )
        return html_response(
            page(title="Reasoner playground", breadcrumbs=_BREADCRUMB_ROOT, body=table_html)
        )

    @router.get("/playground/{day}", include_in_schema=False)
    async def playground_day(
        day: str, request: Request, _: None = Depends(require_session),
    ) -> HTMLResponse:
        if not _DAY_RE.match(day):
            return _not_found(_BREADCRUMB_ROOT, "No such day.")
        state, day_path = _resolve_day(trace_dir, day)
        if state != "ok":
            return html_response(
                page(
                    title=f"Playground: {day}",
                    breadcrumbs=_BREADCRUMB_ROOT + [(day, f"/admin/playground/{day}")],
                    body=_empty_state_body(state),
                )
            )
        records = _load_day_records(day_path)
        if not records:
            return html_response(
                page(
                    title=f"Playground: {day}",
                    breadcrumbs=_BREADCRUMB_ROOT + [(day, f"/admin/playground/{day}")],
                    body=_empty_state_body("empty"),
                )
            )
        query = request.query_params
        clamped_offset = clamp_offset(query.get("offset"))
        clamped_limit = clamp_limit(query.get("limit"))
        page_records = records[clamped_offset: clamped_offset + clamped_limit]
        rows = [
            [
                entry["local_time"],
                entry.get("label", "-"),
                entry.get("model", "-"),
                entry.get("tier", "-"),
                entry.get("status", "-"),
                entry.get("estimated_tokens", "-"),
                "yes" if entry.get("error") else "no",
            ]
            for entry in page_records
        ]
        row_hrefs = [f"/admin/playground/{day}/{entry['name']}" for entry in page_records]
        base_path = f"/admin/playground/{day}"
        table_html = table_component(
            headers=["Local time", "Label", "Model", "Tier", "Status", "Est. tokens", "Error?"],
            rows=rows,
            column_classes=["nowrap mono", "nowrap", "nowrap", "num nowrap", "num nowrap",
                            "num nowrap", "nowrap"],
            row_hrefs=row_hrefs,
            offset=clamped_offset,
            limit=clamped_limit,
            total=len(records),
            base_path=base_path,
        )
        breadcrumbs = _BREADCRUMB_ROOT + [(day, base_path)]
        return html_response(
            page(title=f"Playground: {day}", breadcrumbs=breadcrumbs, body=table_html)
        )

    @router.get("/playground/{day}/{name}", include_in_schema=False)
    async def playground_call(
        day: str, name: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        if not _DAY_RE.match(day) or not _NAME_RE.match(name):
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        state, day_path = _resolve_day(trace_dir, day)
        if state != "ok":
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        target = _safe_join(day_path, name)
        if target is None or not target.is_file():
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        breadcrumbs = _BREADCRUMB_ROOT + [
            (day, f"/admin/playground/{day}"),
            (name, f"/admin/playground/{day}/{name}"),
        ]
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return html_response(
                page(
                    title=f"Playground call: {name}",
                    breadcrumbs=breadcrumbs,
                    body="<p>This trace file could not be parsed as JSON.</p>",
                ),
                status_code=200,
            )
        form_html: str | None = None
        if enabled and model is not None and csrf_token is not None:
            form_html = _replay_form_html(
                record,
                action=f"/admin/playground/{day}/{name}/run",
                csrf_value=csrf_token(),
            )
        body = _render_call_detail(record, form_html=form_html)
        return html_response(
            page(title=f"Playground call: {name}", breadcrumbs=breadcrumbs, body=body)
        )

    @router.post("/playground/{day}/{name}/run", include_in_schema=False)
    async def playground_run(
        day: str, name: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        if not enabled or model is None:
            return _not_found(_BREADCRUMB_ROOT, "Playground replay is disabled.")
        if not _DAY_RE.match(day) or not _NAME_RE.match(name):
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        state, day_path = _resolve_day(trace_dir, day)
        if state != "ok":
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        target = _safe_join(day_path, name)
        if target is None or not target.is_file():
            return _not_found(_BREADCRUMB_ROOT, "No such call.")

        breadcrumbs = _BREADCRUMB_ROOT + [
            (day, f"/admin/playground/{day}"),
            (name, f"/admin/playground/{day}/{name}"),
        ]

        raw_body = (await request.body()).decode("utf-8", errors="replace")
        fields = dict(parse_qsl(raw_body, keep_blank_values=True))

        if verify_csrf is None or not verify_csrf(fields.get("csrf_token")):
            return html_response(
                page(
                    title="Forbidden", breadcrumbs=breadcrumbs,
                    body="<p>Missing or invalid CSRF token.</p>",
                ),
                status_code=403,
            )

        try:
            original_label = json.loads(target.read_text(encoding="utf-8")).get(
                "label", "replay"
            )
        except (OSError, ValueError):
            original_label = "replay"

        try:
            chat_request = _parse_replay_form(fields)
        except (ValueError, ValidationError) as error:
            return html_response(
                page(
                    title="Replay failed", breadcrumbs=breadcrumbs,
                    body=f"<p>Could not build the replay request: {esc(error)}</p>",
                )
            )

        label = f"playground-{original_label}"
        tier = TIER_FOREGROUND
        try:
            with llm_call_tier(tier, label):
                completion = await model.complete(chat_request, trace=False)
            status_code = 200
            error_text = None
        except (LLMError, ValueError) as error:
            completion = None
            status_code = 0
            error_text = str(error)

        sequence = next(_playground_sequence)
        record = build_trace_record(
            sequence=sequence,
            label=label,
            tier=tier,
            model=chat_request.model,
            status=status_code,
            estimated_tokens=model.context_budget.estimate_request(chat_request),
            request=chat_request,
            completion=completion,
            error=error_text,
        )
        playground_root = trace_dir / "playground" if trace_dir is not None else None
        if playground_root is None:
            return html_response(
                page(
                    title="Replay failed", breadcrumbs=breadcrumbs,
                    body="<p>No trace directory is configured; the replay cannot be recorded.</p>",
                )
            )
        written = write_trace_file(playground_root, record, label, sequence)
        if written is None:
            return html_response(
                page(
                    title="Replay failed", breadcrumbs=breadcrumbs,
                    body="<p>Could not write the replay trace file.</p>",
                )
            )
        replay_day = written.parent.name
        replay_name = written.name
        return RedirectResponse(
            url=f"/admin/playground/{day}/{name}/compare/{replay_day}/{replay_name}",
            status_code=303,
        )

    @router.get(
        "/playground/{day}/{name}/compare/{replay_day}/{replay_name}",
        include_in_schema=False,
    )
    async def playground_compare(
        day: str, name: str, replay_day: str, replay_name: str,
        _: None = Depends(require_session),
    ) -> HTMLResponse:
        if not _DAY_RE.match(day) or not _NAME_RE.match(name):
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        if not _DAY_RE.match(replay_day) or not _NAME_RE.match(replay_name):
            return _not_found(_BREADCRUMB_ROOT, "No such replay.")
        state, day_path = _resolve_day(trace_dir, day)
        if state != "ok":
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        target = _safe_join(day_path, name)
        if target is None or not target.is_file():
            return _not_found(_BREADCRUMB_ROOT, "No such call.")
        replay_state, replay_day_path = _resolve_day(_playground_root(), replay_day)
        if replay_state != "ok":
            return _not_found(_BREADCRUMB_ROOT, "No such replay.")
        replay_target = _safe_join(replay_day_path, replay_name)
        if replay_target is None or not replay_target.is_file():
            return _not_found(_BREADCRUMB_ROOT, "No such replay.")

        try:
            original = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            original = {}
        try:
            replay = json.loads(replay_target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            replay = {}

        body = (
            '<div class="playground-compare">'
            f"<section><h2>Original ({esc(name)})</h2>"
            f"{_render_call_detail(original, include_replay_section=False)}</section>"
            f"<section><h2>Replay ({esc(replay_name)})</h2>"
            f"{_render_call_detail(replay, include_replay_section=False)}</section>"
            "</div>"
        )
        breadcrumbs = _BREADCRUMB_ROOT + [
            (day, f"/admin/playground/{day}"),
            (name, f"/admin/playground/{day}/{name}"),
            ("Compare", ""),
        ]
        return html_response(page(title=f"Compare: {name}", breadcrumbs=breadcrumbs, body=body))


def _not_found(breadcrumbs: list[tuple[str, str]], message: str) -> HTMLResponse:
    return html_response(
        page(title="Not found", breadcrumbs=breadcrumbs, body=f"<p>{esc(message)}</p>"),
        status_code=404,
    )


def _empty_state_body(state: str) -> str:
    if state == "unset":
        return (
            "<p>LLM call tracing is not configured. Set "
            "<code>GOSSIPMEMO_LLM_TRACE_PATH</code> to a directory and restart "
            "the server to browse past calls here.</p>"
        )
    if state == "missing":
        return "<p>The configured trace directory does not exist yet. No calls have been traced.</p>"
    return "<p>No calls traced on this day.</p>"


def _list_days(root: Path | None) -> tuple[str, list[tuple[str, int]]]:
    """Return `(state, days)`. `state` is `"ok"`, `"unset"`, or `"missing"`.

    `days` is `[(name, call_count), ...]`, newest first -- the `YYYY-MM-DD`
    name sorts lexicographically the same as chronologically, so a plain
    reverse sort suffices. `root` is either `trace_dir` (production calls)
    or `trace_dir / "playground"` (replay history) -- this helper does not
    care which.
    """

    if root is None:
        return "unset", []
    try:
        entries = list(root.iterdir())
    except OSError:
        return "missing", []
    days: list[tuple[str, int]] = []
    for entry in entries:
        if not entry.is_dir() or not _DAY_RE.match(entry.name):
            continue
        try:
            file_count = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix == ".json")
        except OSError:
            file_count = 0
        days.append((entry.name, file_count))
    days.sort(key=lambda item: item[0], reverse=True)
    return "ok", days


def _resolve_day(root: Path | None, day: str) -> tuple[str, Path | None]:
    """Validate `day` already matched `_DAY_RE`; resolve it under `root`
    and confirm it stays inside as a second line of defense."""

    if root is None:
        return "unset", None
    day_path = _safe_join(root, day)
    if day_path is None or not day_path.is_dir():
        return "missing", None
    return "ok", day_path


def _safe_join(base: Path, child: str) -> Path | None:
    """Join `child` onto `base` and confirm the result stays inside `base`.

    Callers must already have validated `child` against a strict regex --
    this is a second line of defense, not the only one, since a resolved
    symlink or a `base` that itself doesn't exist yet can behave
    surprisingly. Returns `None` rather than raising on any of that.
    """

    try:
        base_resolved = base.resolve()
        candidate = (base / child).resolve()
    except OSError:
        return None
    if candidate != base_resolved and base_resolved not in candidate.parents:
        return None
    return candidate


def _load_day_records(day_path: Path) -> list[dict]:
    """Load every trace file in `day_path`, newest first.

    A file that fails to parse is shown as an error row rather than
    raising -- this is meant to survive a partially-written or corrupt
    trace file, not just the happy path.
    """

    try:
        names = sorted(
            (f.name for f in day_path.iterdir() if f.is_file() and _NAME_RE.match(f.name)),
            reverse=True,
        )
    except OSError:
        return []
    records: list[dict] = []
    for name in names:
        local_time = _format_local_time(name)
        entry: dict = {"name": name, "local_time": local_time}
        try:
            record = json.loads((day_path / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entry["label"] = "(unparseable trace file)"
            entry["status"] = "error"
            entry["error"] = "unparseable"
        else:
            entry["label"] = record.get("label", "-")
            entry["model"] = record.get("model", "-")
            entry["tier"] = record.get("tier", "-")
            entry["status"] = record.get("status", "-")
            entry["estimated_tokens"] = record.get("estimated_tokens", "-")
            entry["error"] = record.get("error")
        records.append(entry)
    return records


def _format_local_time(name: str) -> str:
    """`HHMMSSmmm` from the filename prefix, rendered as `HH:MM:SS.mmm`."""

    prefix = name[:9]
    return f"{prefix[0:2]}:{prefix[2:4]}:{prefix[4:6]}.{prefix[6:9]}"


def _render_call_detail(
    record: dict, *, form_html: str | None = None, include_replay_section: bool = True,
) -> str:
    meta_rows = [
        ("Sequence", record.get("sequence", "-")),
        ("Timestamp (UTC)", record.get("timestamp", "-")),
        ("Label", record.get("label", "-")),
        ("Tier", record.get("tier", "-")),
        ("Model", record.get("model", "-")),
        ("Status", record.get("status", "-")),
        ("Estimated tokens", record.get("estimated_tokens", "-")),
    ]
    meta_html = "<ul>" + "".join(
        f"<li>{esc(label)}: {esc(value)}</li>" for label, value in meta_rows
    ) + "</ul>"
    sections = [f"<section><h2>Call</h2>{meta_html}</section>"]

    request = record.get("request") or {}
    messages = request.get("messages") or []
    message_blocks = (
        "".join(
            _message_block(
                message.get("role", "?") if isinstance(message, dict) else "?",
                message.get("content", "") if isinstance(message, dict) else message,
            )
            for message in messages
        )
        if messages
        else "<p>(no messages)</p>"
    )
    sections.append(f"<section><h2>Request messages</h2>{message_blocks}</section>")

    completion = record.get("completion")
    completion_html = f"<pre>{esc(completion)}</pre>" if completion else "<p>(none)</p>"
    sections.append(f"<section><h2>Completion</h2>{completion_html}</section>")

    error = record.get("error")
    error_html = f"<pre>{esc(error)}</pre>" if error else "<p>(none)</p>"
    sections.append(f"<section><h2>Error</h2>{error_html}</section>")

    if include_replay_section:
        if form_html:
            sections.append(f"<section><h2>Replay</h2>{form_html}</section>")
        else:
            sections.append(
                "<section><h2>Replay</h2><p>The playground replay form is disabled. Set "
                "<code>GOSSIPMEMO_ADMIN_PLAYGROUND_ENABLED=true</code> to enable it &mdash; "
                "doing so lets the admin UI spend real money on LLM calls.</p></section>"
            )

    return "\n".join(sections)


def _message_block(role: object, content: object) -> str:
    if isinstance(content, (list, dict)):
        content_text = json.dumps(content, indent=2, ensure_ascii=False)
    else:
        content_text = "" if content is None else str(content)
    return f"<h3>{esc(role)}</h3><blockquote><pre>{esc(content_text)}</pre></blockquote>"


def _replay_form_html(record: dict, *, action: str, csrf_value: str) -> str:
    """Build the replay form, prefilled from `record["request"]`.

    Zero JavaScript: a fixed set of message textareas matching the
    record's own messages, no add/remove-row controls. Every prefilled
    value goes through `esc()`.
    """

    request = record.get("request") or {}
    messages = request.get("messages") or []
    model_value = request.get("model", "")
    temperature = request.get("temperature")
    max_tokens = request.get("max_tokens")
    response_format = request.get("response_format")
    response_format_type = (
        response_format.get("type", "") if isinstance(response_format, dict) else ""
    )

    message_fields: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role", "") if isinstance(message, dict) else ""
        content = message.get("content", "") if isinstance(message, dict) else message
        if isinstance(content, (list, dict)):
            content_text = json.dumps(content, indent=2, ensure_ascii=False)
        else:
            content_text = "" if content is None else str(content)
        message_fields.append(
            f'<div class="playground-message">'
            f'<label for="message_{index}_content">Message {index} ({esc(role)})</label>'
            f'<input type="hidden" name="message_{index}_role" value="{esc(role)}">'
            f'<textarea id="message_{index}_content" name="message_{index}_content" '
            f'rows="14" cols="100">{esc(content_text)}</textarea>'
            f"</div>"
        )

    return f"""
<p>Edits here are wire-level only: a tuned prompt must be ported back into
<code>prompts/defaults.py</code> or a <code>prompts.toml</code> override by hand to
actually ship &mdash; submitting this form does not change what any reasoner sends.</p>
<form method="post" action="{esc(action)}">
<input type="hidden" name="csrf_token" value="{esc(csrf_value)}">
<input type="hidden" name="message_count" value="{len(messages)}">
<label for="model">Model</label>
<input type="text" id="model" name="model" value="{esc(model_value)}">
<label for="temperature">Temperature</label>
<input type="text" id="temperature" name="temperature"
value="{esc('' if temperature is None else temperature)}">
<label for="max_tokens">Max tokens</label>
<input type="text" id="max_tokens" name="max_tokens"
value="{esc('' if max_tokens is None else max_tokens)}">
<label for="response_format">Response format type (blank, or e.g. "json_object")</label>
<input type="text" id="response_format" name="response_format" value="{esc(response_format_type)}">
{"".join(message_fields)}
<button type="submit">Run against configured LLM</button>
</form>
"""


def _parse_replay_form(fields: dict[str, str]) -> ChatCompletionRequest:
    """Build the `ChatCompletionRequest` the replay form describes.

    Raises `ValueError` or `pydantic.ValidationError` on anything
    malformed; the caller renders those as an error page rather than
    letting them turn into a 500.
    """

    model_value = fields.get("model", "").strip()
    if not model_value:
        raise ValueError("model must not be empty")

    temperature_raw = fields.get("temperature", "").strip()
    temperature = float(temperature_raw) if temperature_raw else None

    max_tokens_raw = fields.get("max_tokens", "").strip()
    max_tokens = int(max_tokens_raw) if max_tokens_raw else None

    response_format_raw = fields.get("response_format", "").strip()
    response_format = {"type": response_format_raw} if response_format_raw else None

    try:
        message_count = int(fields.get("message_count", "0"))
    except ValueError as error:
        raise ValueError("malformed message_count") from error
    if message_count < 0:
        raise ValueError("malformed message_count")

    messages: list[ChatMessage] = []
    for index in range(message_count):
        role = fields.get(f"message_{index}_role", "user")
        content = fields.get(f"message_{index}_content", "")
        messages.append(ChatMessage(role=role, content=content))
    if not messages:
        raise ValueError("at least one message is required")

    return ChatCompletionRequest(
        model=model_value,
        messages=messages,
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens,
    )


__all__ = ["register"]
