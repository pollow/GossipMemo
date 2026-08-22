"""Read-only admin views: the reasoner playground.

Renders the per-call JSON trace files `OpenAICompatibleAdapter._trace`
writes under `Settings.llm_trace_path` (see `src/gossipmemo/llm.py`):

    <trace_dir>/<YYYY-MM-DD>/<HHMMSSmmm>-<label>-<sequence>.json

This slice is strictly a browser: three GET routes, no POST, no write of
any kind, no LLM call. A later slice adds an editable form that re-runs a
call and writes its replay traces under a reserved `<trace_dir>/playground/`
subdirectory -- `_DAY_RE` below deliberately does not match that name, so
it never shows up in the day list.

`day` and `name` arrive from the URL and would otherwise be joined onto a
filesystem path taken from settings. Both are validated against a strict
regex *before* any `Path` join, and the resulting path is additionally
resolved and checked to still be inside the trace directory, as a second
line of defense against traversal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..render import esc, html_response, page, table_component
from ._common import clamp_limit, clamp_offset

#: Day directories only. Excludes the reserved `playground/` subdirectory
#: slice 3 uses for replay traces, and anything else that isn't a day.
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: `<HHMMSSmmm>-<label>-<sequence>[-<attempt>].json`, matching exactly what
#: `_trace` writes (see `llm.py`): a 9-digit local-time prefix, a label
#: reduced to `[a-z0-9-]`, a 4-digit sequence, and an optional numeric
#: collision suffix.
_NAME_RE = re.compile(r"^\d{9}-[a-z0-9-]+-\d{4}(-\d+)?\.json$")

_BREADCRUMB_ROOT = [("Admin", "/admin"), ("Playground", "/admin/playground")]


def register(router: APIRouter, require_session, trace_dir: Path | None) -> None:
    """Attach the playground routes to `router`.

    `trace_dir` is `Settings.llm_trace_path`, passed in rather than
    re-read from settings so this module never imports `Settings`.
    """

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
        rows = [[day, count] for day, count in days]
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
            return _not_found("No such day.")
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
            return _not_found("No such call.")
        state, day_path = _resolve_day(trace_dir, day)
        if state != "ok":
            return _not_found("No such call.")
        target = _safe_join(day_path, name)
        if target is None or not target.is_file():
            return _not_found("No such call.")
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return html_response(
                page(
                    title=f"Playground call: {name}",
                    breadcrumbs=_BREADCRUMB_ROOT
                    + [(day, f"/admin/playground/{day}"), (name, "")],
                    body="<p>This trace file could not be parsed as JSON.</p>",
                ),
                status_code=200,
            )
        body = _render_call_detail(record)
        breadcrumbs = _BREADCRUMB_ROOT + [
            (day, f"/admin/playground/{day}"),
            (name, f"/admin/playground/{day}/{name}"),
        ]
        return html_response(
            page(title=f"Playground call: {name}", breadcrumbs=breadcrumbs, body=body)
        )


def _not_found(message: str) -> HTMLResponse:
    return html_response(
        page(title="Not found", breadcrumbs=_BREADCRUMB_ROOT, body=f"<p>{esc(message)}</p>"),
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


def _list_days(trace_dir: Path | None) -> tuple[str, list[tuple[str, int]]]:
    """Return `(state, days)`. `state` is `"ok"`, `"unset"`, or `"missing"`.

    `days` is `[(name, call_count), ...]`, newest first -- the `YYYY-MM-DD`
    name sorts lexicographically the same as chronologically, so a plain
    reverse sort suffices.
    """

    if trace_dir is None:
        return "unset", []
    try:
        entries = list(trace_dir.iterdir())
    except OSError:
        return "missing", []
    days: list[tuple[str, int]] = []
    for entry in entries:
        if not entry.is_dir() or not _DAY_RE.match(entry.name):
            continue
        try:
            count = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix == ".json")
        except OSError:
            count = 0
        days.append((entry.name, count))
    days.sort(key=lambda item: item[0], reverse=True)
    return "ok", days


def _resolve_day(trace_dir: Path | None, day: str) -> tuple[str, Path | None]:
    """Validate `day` already matched `_DAY_RE`; resolve it under `trace_dir`
    and confirm it stays inside as a second line of defense."""

    if trace_dir is None:
        return "unset", None
    day_path = _safe_join(trace_dir, day)
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


def _render_call_detail(record: dict) -> str:
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

    return "\n".join(sections)


def _message_block(role: object, content: object) -> str:
    if isinstance(content, (list, dict)):
        content_text = json.dumps(content, indent=2, ensure_ascii=False)
    else:
        content_text = "" if content is None else str(content)
    return f"<h3>{esc(role)}</h3><blockquote><pre>{esc(content_text)}</pre></blockquote>"


__all__ = ["register"]
