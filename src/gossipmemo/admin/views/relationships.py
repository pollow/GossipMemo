"""Read-only admin views: relationship list and relationship dossier.

The dossier route reuses `store.relationship_context`, the same read the
public `/v1/spaces/{sid}/relationships/{rid}` endpoint uses.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ...store.sqlite import SqliteWorldStore
from ..render import esc, html_response, page, table_component
from ._common import clamp_limit, clamp_offset, require_space, space_breadcrumbs


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/spaces/{space_id}/relationships", include_in_schema=False)
    async def relationships_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        total = store.admin_count_relationships(space_id)
        rows = store.admin_list_relationships(space_id, offset, limit)
        base_path = f"/admin/spaces/{space_id}/relationships"
        table_html = table_component(
            headers=["Person A", "Person B", "Status", "Summary"],
            rows=[
                [
                    row.person_a_name,
                    row.person_b_name,
                    row.status,
                    (row.summary[:120] + "...") if len(row.summary) > 120 else row.summary,
                ]
                for row in rows
            ],
            row_hrefs=[f"{base_path}/{row.id}" for row in rows],
            offset=offset,
            limit=limit,
            total=total,
            base_path=base_path,
        )
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("Relationships", base_path)
        ]
        return html_response(
            page(title=f"Relationships: {overview.name}", breadcrumbs=breadcrumbs, body=table_html)
        )

    @router.get("/spaces/{space_id}/relationships/{relationship_id}", include_in_schema=False)
    async def relationship_detail_view(
        space_id: str, relationship_id: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        base_path = f"/admin/spaces/{space_id}/relationships"
        context = store.relationship_context(space_id, relationship_id)
        if context is None:
            breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
                ("Relationships", base_path),
                (relationship_id, f"{base_path}/{relationship_id}"),
            ]
            return html_response(
                page(title="Relationship not found", breadcrumbs=breadcrumbs,
                     body="<p>No such relationship.</p>"),
                status_code=404,
            )
        relationship, memories, watermark = context
        person_a_name = (
            store.admin_person_display_name(space_id, relationship.person_a_id)
            or relationship.person_a_id
        )
        person_b_name = (
            store.admin_person_display_name(space_id, relationship.person_b_id)
            or relationship.person_b_id
        )
        body = _render_relationship_detail(
            space_id, relationship, person_a_name, person_b_name, memories, watermark
        )
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("Relationships", base_path),
            (relationship_id, f"{base_path}/{relationship_id}"),
        ]
        return html_response(
            page(
                title=f"Relationship {relationship_id}",
                breadcrumbs=breadcrumbs,
                body=body,
            )
        )


def _render_relationship_detail(
    space_id: str, relationship, person_a_name: str, person_b_name: str, memories, watermark
) -> str:
    memories_html = (
        "<ul>" + "".join(
            f'<li><a href="/admin/spaces/{esc(space_id)}/memories/{esc(m.id)}">'
            f"{esc(m.content[:120])}</a> ({esc(m.kind)}, {esc(m.status)})</li>"
            for m in memories
        ) + "</ul>"
        if memories
        else "<p>None linked.</p>"
    )
    return f"""
<section>
<h2>Relationship</h2>
<ul>
<li>Id: {esc(relationship.id)}</li>
<li>Person A: <a href="/admin/spaces/{esc(space_id)}/people/{esc(relationship.person_a_id)}">{esc(person_a_name)}</a></li>
<li>Person B: <a href="/admin/spaces/{esc(space_id)}/people/{esc(relationship.person_b_id)}">{esc(person_b_name)}</a></li>
<li>Status: {esc(relationship.status)}</li>
<li>Closeness: {esc(relationship.closeness or "-")}</li>
<li>Tone: {esc(relationship.tone or "-")}</li>
<li>Stale: {esc("yes" if relationship.stale else "no")}</li>
<li>Profile updated: {esc(relationship.profile_updated_at or "never")}</li>
<li>Source watermark: {esc(watermark or "none")}</li>
</ul>
<p>{esc(relationship.summary or "(no summary)")}</p>
</section>
<section>
<h2>Linked memories</h2>
{memories_html}
</section>
"""


__all__ = ["register"]
