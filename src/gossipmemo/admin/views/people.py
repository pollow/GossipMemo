"""Read-only admin views: people list and person dossier.

The dossier route reuses `store.person_context`, the same read the public
`/v1/spaces/{sid}/people/{pid}` endpoint uses, rather than re-deriving the
card/linked-memories assembly here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ...store.sqlite import SqliteWorldStore
from ..render import esc, html_response, json_block, page, table_component
from ._common import clamp_limit, clamp_offset, require_space, space_breadcrumbs


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/spaces/{space_id}/people", include_in_schema=False)
    async def people_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        total = store.admin_count_people(space_id)
        rows = store.admin_list_people(space_id, offset, limit)
        base_path = f"/admin/spaces/{space_id}/people"
        table_html = table_component(
            headers=["Display name", "Aliases", "Status"],
            rows=[
                [row.display_name, ", ".join(row.aliases) or "(none)", row.status]
                for row in rows
            ],
            row_hrefs=[f"{base_path}/{row.id}" for row in rows],
            offset=offset,
            limit=limit,
            total=total,
            base_path=base_path,
        )
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [("People", base_path)]
        return html_response(
            page(title=f"People: {overview.name}", breadcrumbs=breadcrumbs, body=table_html)
        )

    @router.get("/spaces/{space_id}/people/{person_id}", include_in_schema=False)
    async def person_detail_view(
        space_id: str, person_id: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        base_path = f"/admin/spaces/{space_id}/people"
        context = store.person_context(space_id, person_id)
        if context is None:
            breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
                ("People", base_path),
                (person_id, f"{base_path}/{person_id}"),
            ]
            return html_response(
                page(title="Person not found", breadcrumbs=breadcrumbs,
                     body="<p>No such person.</p>"),
                status_code=404,
            )
        person, memories, watermark = context
        aliases = store.admin_person_aliases(space_id, person_id)
        body = _render_person_detail(space_id, person, aliases, memories, watermark)
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("People", base_path),
            (person.display_name, f"{base_path}/{person_id}"),
        ]
        return html_response(
            page(title=f"Person: {person.display_name}", breadcrumbs=breadcrumbs, body=body)
        )


def _render_person_detail(space_id: str, person, aliases: list[str], memories, watermark) -> str:
    aliases_html = (
        "<ul>" + "".join(f"<li>{esc(a)}</li>" for a in aliases) + "</ul>"
        if aliases
        else "<p>None.</p>"
    )
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
<h2>Profile</h2>
<ul>
<li>Id: {esc(person.id)}</li>
<li>Stale: {esc("yes" if person.stale else "no")}</li>
<li>Profile updated: {esc(person.profile_updated_at or "never")}</li>
<li>Source watermark: {esc(watermark or "none")}</li>
</ul>
{json_block(person.profile_card)}
</section>
<section>
<h2>Aliases</h2>
{aliases_html}
</section>
<section>
<h2>Linked memories</h2>
{memories_html}
</section>
"""


__all__ = ["register"]
