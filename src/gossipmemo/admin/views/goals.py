"""Read-only admin view: learning goals, filterable by coverage root and status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ...models import COVERAGE_ROOTS
from ...store.sqlite import SqliteWorldStore
from ..render import esc, html_response, page, table_component
from ._common import clamp_limit, clamp_offset, clean_choice, require_space, space_breadcrumbs

_GOAL_STATUSES = ("open", "partial", "answered", "deferred", "retired")


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/spaces/{space_id}/goals", include_in_schema=False)
    async def goals_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        root = clean_choice(query.get("root"), COVERAGE_ROOTS)
        status = clean_choice(query.get("status"), _GOAL_STATUSES)

        total = store.admin_count_learning_goals(space_id, root=root, status=status)
        rows = store.admin_list_learning_goals(space_id, offset, limit, root=root, status=status)
        extra_params: dict[str, str] = {}
        if root:
            extra_params["root"] = root
        if status:
            extra_params["status"] = status

        base_path = f"/admin/spaces/{space_id}/goals"
        table_html = table_component(
            headers=["Prompt", "Focus", "Status", "Updated at"],
            rows=[
                [
                    row.prompt,
                    f"{row.focus_kind}" + (f" ({row.focus_id})" if row.focus_id else ""),
                    row.status,
                    row.updated_at,
                ]
                for row in rows
            ],
            column_classes=["wrap", "nowrap", "nowrap", "nowrap mono"],
            offset=offset,
            limit=limit,
            total=total,
            base_path=base_path,
            extra_params=extra_params,
        )
        filter_form = _goal_filter_form(base_path, root, status)
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [("Learning goals", base_path)]
        return html_response(
            page(
                title=f"Learning goals: {overview.name}",
                breadcrumbs=breadcrumbs,
                body=filter_form + _goal_rationales(rows) + table_html,
            )
        )


def _goal_rationales(rows) -> str:
    if not rows:
        return ""
    items = "".join(
        f"<li><strong>{esc(row.prompt)}</strong>: {esc(row.rationale)}"
        f"{' -- ' + esc(row.status_reason) if row.status_reason else ''}</li>"
        for row in rows
    )
    return f"<section><h2>Rationale</h2><ul>{items}</ul></section>"


def _goal_filter_form(base_path: str, root: str | None, status: str | None) -> str:
    def option(value: str, label: str, selected: str | None) -> str:
        is_selected = " selected" if value == (selected or "") else ""
        return f'<option value="{esc(value)}"{is_selected}>{esc(label)}</option>'

    root_options = "".join(
        [option("", "Any", root)] + [option(value, value, root) for value in COVERAGE_ROOTS]
    )
    status_options = "".join(
        [option("", "Any", status)]
        + [option(value, value, status) for value in _GOAL_STATUSES]
    )
    return f"""
<form method="get" action="{esc(base_path)}">
<label for="root">Coverage root</label>
<select id="root" name="root">{root_options}</select>
<label for="status">Status</label>
<select id="status" name="status">{status_options}</select>
<button type="submit">Filter</button>
</form>
"""


__all__ = ["register"]
