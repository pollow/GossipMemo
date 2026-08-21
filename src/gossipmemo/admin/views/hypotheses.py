"""Read-only admin view: hypotheses, filterable by owner kind, with their
support/counter evidence shown distinctly."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ...store._admin import HypothesisRow
from ...store.sqlite import SqliteWorldStore
from ..render import esc, html_response, page, table_component
from ._common import clamp_limit, clamp_offset, clean_choice, require_space, space_breadcrumbs

_OWNER_KINDS = ("user", "person", "relationship")


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/spaces/{space_id}/hypotheses", include_in_schema=False)
    async def hypotheses_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        owner_kind = clean_choice(query.get("owner_kind"), _OWNER_KINDS)

        total = store.admin_count_hypotheses(space_id, owner_kind=owner_kind)
        rows = store.admin_list_hypotheses(space_id, offset, limit, owner_kind=owner_kind)
        extra_params: dict[str, str] = {}
        if owner_kind:
            extra_params["owner_kind"] = owner_kind

        base_path = f"/admin/spaces/{space_id}/hypotheses"
        table_html = table_component(
            headers=["Content", "Owner", "Kind", "Confidence", "Status"],
            rows=[
                [
                    row.content,
                    f"{row.owner_kind}" + (f" ({row.owner_id})" if row.owner_id else ""),
                    row.kind,
                    row.confidence,
                    row.status,
                ]
                for row in rows
            ],
            offset=offset,
            limit=limit,
            total=total,
            base_path=base_path,
            extra_params=extra_params,
        )
        filter_form = _owner_kind_filter_form(base_path, owner_kind)
        evidence_html = _evidence_sections(space_id, rows)
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [("Hypotheses", base_path)]
        return html_response(
            page(
                title=f"Hypotheses: {overview.name}",
                breadcrumbs=breadcrumbs,
                body=filter_form + table_html + evidence_html,
            )
        )


def _owner_kind_filter_form(base_path: str, owner_kind: str | None) -> str:
    def option(value: str, label: str, selected: str | None) -> str:
        is_selected = " selected" if value == (selected or "") else ""
        return f'<option value="{esc(value)}"{is_selected}>{esc(label)}</option>'

    owner_kind_options = "".join(
        [option("", "Any", owner_kind)]
        + [option(value, value, owner_kind) for value in _OWNER_KINDS]
    )
    return f"""
<form method="get" action="{esc(base_path)}">
<label for="owner_kind">Owner kind</label>
<select id="owner_kind" name="owner_kind">{owner_kind_options}</select>
<button type="submit">Filter</button>
</form>
"""


def _evidence_sections(space_id: str, rows: list[HypothesisRow]) -> str:
    if not rows:
        return ""
    items = "".join(_evidence_section(space_id, row) for row in rows)
    return f"<section><h2>Evidence</h2>{items}</section>"


def _evidence_section(space_id: str, row: HypothesisRow) -> str:
    def evidence_list(evidence) -> str:
        if not evidence:
            return "<p>None.</p>"
        return "<ul>" + "".join(
            f'<li><a href="/admin/spaces/{esc(space_id)}/memories/{esc(item.memory_id)}">'
            f"{esc(item.content[:100])}</a></li>"
            for item in evidence
        ) + "</ul>"

    return f"""
<h3>{esc(row.content[:100])}</h3>
<p>Support</p>
{evidence_list(row.support_evidence)}
<p>Counter</p>
{evidence_list(row.counter_evidence)}
"""


__all__ = ["register"]
