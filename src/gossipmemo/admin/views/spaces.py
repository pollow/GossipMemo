"""Read-only admin views: spaces, messages, memories.

Every route is session-protected and only ever issues GET requests against
`_AdminReadMixin` (see `store/_admin.py`). No route here writes anything.
Filters and pagination are parsed from query strings by hand and clamped
rather than trusted, then passed to the store as bound parameters -- never
interpolated into SQL.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ...store._admin import MemoryDetail, MessageRow
from ...store.sqlite import SqliteWorldStore
from ..render import esc, html_response, json_block, page, table_component
from ._common import (
    clamp_limit,
    clamp_offset,
    clean_bool,
    clean_choice,
    require_space,
    space_breadcrumbs,
)

_MEMORY_STATES = ("active", "retracted", "superseded")
_MEMORY_KINDS = ("fact", "event", "preference", "plan", "situation", "impression")


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    """Attach the space/message/memory views to `router`.

    `require_session` is `AdminAuth.require_session`, passed in rather than
    imported so this module never constructs its own `AdminAuth`.
    """

    @router.get("", include_in_schema=False)
    async def landing(_: None = Depends(require_session)) -> HTMLResponse:
        spaces = store.admin_list_spaces()
        if len(spaces) == 1:
            return RedirectResponse(url=f"/admin/spaces/{spaces[0].id}", status_code=303)
        return html_response(_render_space_list(spaces))

    @router.get("/spaces", include_in_schema=False)
    async def space_list(_: None = Depends(require_session)) -> HTMLResponse:
        spaces = store.admin_list_spaces()
        return html_response(_render_space_list(spaces))

    @router.get("/spaces/{space_id}", include_in_schema=False)
    async def space_overview(
        space_id: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        body = f"""
<section>
<h2>Search</h2>
<form method="get" action="/admin/spaces/{esc(space_id)}/search">
<label for="q">Search this space</label>
<input type="text" id="q" name="q">
<button type="submit">Search</button>
</form>
</section>
<section>
<h2>Contents</h2>
{_render_stats(space_id, overview)}
</section>
<section>
<h2>User model</h2>
<p>Updated: {esc(overview.user_model_updated_at or "never")}</p>
{json_block(overview.user_model_profile_card)}
</section>
<section>
<h2>Continuity</h2>
<p>Updated: {esc(overview.continuity_updated_at or "never")}</p>
<p>{esc(overview.continuity_text or "(empty)")}</p>
<h3>Related people</h3>
{_render_continuity_people(space_id, overview.continuity_related_people)}
<h3>Last covered message</h3>
{_render_continuity_message(overview.continuity_through_message)}
</section>
"""
        return html_response(
            page(
                title=f"Space: {overview.name}",
                breadcrumbs=space_breadcrumbs(space_id, overview.name),
                body=body,
            )
        )

    @router.get("/spaces/{space_id}/messages", include_in_schema=False)
    async def messages_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        total = store.admin_count_messages(space_id)
        rows = store.admin_list_messages(space_id, offset, limit)
        body = table_component(
            headers=["Occurred at", "Author", "Content", "Extraction batch", "Extracted?"],
            rows=[_message_table_row(row) for row in rows],
            column_classes=["nowrap mono", "nowrap", "wrap", "mono nowrap", "nowrap"],
            offset=offset,
            limit=limit,
            total=total,
            base_path=f"/admin/spaces/{space_id}/messages",
        )
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("Messages", f"/admin/spaces/{space_id}/messages")
        ]
        return html_response(
            page(title=f"Messages: {overview.name}", breadcrumbs=breadcrumbs, body=body)
        )

    @router.get("/spaces/{space_id}/memories", include_in_schema=False)
    async def memories_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        state = clean_choice(query.get("state"), _MEMORY_STATES)
        kind = clean_choice(query.get("kind"), _MEMORY_KINDS)
        about_user = clean_bool(query.get("about_user"))

        total = store.admin_count_memories(
            space_id, state=state, kind=kind, about_user=about_user
        )
        rows = store.admin_list_memories(
            space_id, offset, limit, state=state, kind=kind, about_user=about_user
        )
        extra_params: dict[str, str] = {}
        if state:
            extra_params["state"] = state
        if kind:
            extra_params["kind"] = kind
        if about_user is not None:
            extra_params["about_user"] = "1" if about_user else "0"

        base_path = f"/admin/spaces/{space_id}/memories"
        table_html = table_component(
            headers=["Content", "Kind", "Status", "About user", "Created at"],
            rows=[
                [
                    (row.content[:300] + "...") if len(row.content) > 300 else row.content,
                    row.kind,
                    row.status,
                    "yes" if row.about_user else "no",
                    row.created_at,
                ]
                for row in rows
            ],
            column_classes=["wrap", "nowrap", "nowrap", "nowrap", "nowrap mono"],
            row_hrefs=[f"{base_path}/{row.id}" for row in rows],
            offset=offset,
            limit=limit,
            total=total,
            base_path=base_path,
            extra_params=extra_params,
        )
        filter_form = _memory_filter_form(base_path, state, kind, about_user)
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("Memories", base_path)
        ]
        return html_response(
            page(
                title=f"Memories: {overview.name}",
                breadcrumbs=breadcrumbs,
                body=filter_form + table_html,
            )
        )

    @router.get("/spaces/{space_id}/memories/{memory_id}", include_in_schema=False)
    async def memory_detail_view(
        space_id: str, memory_id: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        detail = store.admin_memory_detail(space_id, memory_id)
        if detail is None:
            body = "<p>No such memory.</p>"
            breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
                ("Memories", f"/admin/spaces/{space_id}/memories"),
                (memory_id, f"/admin/spaces/{space_id}/memories/{memory_id}"),
            ]
            return html_response(
                page(title="Memory not found", breadcrumbs=breadcrumbs, body=body),
                status_code=404,
            )
        body = _render_memory_detail(space_id, detail)
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("Memories", f"/admin/spaces/{space_id}/memories"),
            (memory_id, f"/admin/spaces/{space_id}/memories/{memory_id}"),
        ]
        return html_response(
            page(title=f"Memory {memory_id}", breadcrumbs=breadcrumbs, body=body)
        )


def _render_space_list(spaces) -> str:
    table_html = table_component(
        headers=["Name", "Messages", "Memories", "People"],
        rows=[[s.name, s.message_count, s.memory_count, s.people_count] for s in spaces],
        column_classes=["", "num nowrap", "num nowrap", "num nowrap"],
        row_hrefs=[f"/admin/spaces/{s.id}" for s in spaces],
        offset=0,
        limit=max(len(spaces), 1),
        total=len(spaces),
        base_path="/admin/spaces",
    )
    tables_link = (
        '<p><a href="/admin/tables">Raw tables</a> &middot; '
        '<a href="/admin/playground">Reasoner playground</a></p>'
    )
    return page(
        title="GossipMemo Admin",
        breadcrumbs=[("Admin", "/admin")],
        body=table_html + tables_link,
    )


def _render_stats(space_id: str, overview) -> str:
    """The space's contents as a row of tiles, each linking into its list.

    `admin_space_overview` only counts messages, memories, and people; the
    other four sections are still worth a tile as a way in, so they show a
    dash instead of a number rather than being demoted to a footnote.
    """

    tiles: list[tuple[str, object, str]] = [
        ("Messages", overview.message_count, "messages"),
        ("Memories", overview.memory_count, "memories"),
        ("People", overview.people_count, "people"),
        ("Relationships", None, "relationships"),
        ("Learning goals", None, "goals"),
        ("Hypotheses", None, "hypotheses"),
        ("Coverage", None, "coverage"),
    ]
    items = "".join(
        f'<li><a href="/admin/spaces/{esc(space_id)}/{esc(path)}">'
        f'<span class="stat-label">{esc(label)}</span>'
        + (
            f'<span class="stat-value">{esc(count)}</span>'
            if count is not None
            else '<span class="stat-value unknown">view</span>'
        )
        + "</a></li>"
        for label, count, path in tiles
    )
    return f'<ul class="stats">{items}</ul>'


def _render_continuity_people(space_id: str, people) -> str:
    if not people:
        return "<p>None linked.</p>"
    return "<ul>" + "".join(
        f'<li><a href="/admin/spaces/{esc(space_id)}/people/{esc(person.id)}">'
        f"{esc(person.display_name)}</a></li>"
        for person in people
    ) + "</ul>"


def _render_continuity_message(message: MessageRow | None) -> str:
    if message is None:
        return "<p>None.</p>"
    return (
        f"<p>[{esc(message.occurred_at)}] {esc(message.author)}: "
        f"{esc(message.content)}</p>"
    )


def _message_table_row(row: MessageRow) -> list[object]:
    extracted = "yes" if row.extraction_state == "completed" else row.extraction_state
    return [
        row.occurred_at,
        row.author,
        (row.content[:400] + "...") if len(row.content) > 400 else row.content,
        row.extraction_batch_id or "(none)",
        extracted,
    ]


def _memory_filter_form(
    base_path: str, state: str | None, kind: str | None, about_user: bool | None
) -> str:
    def option(value: str, label: str, selected: str | None) -> str:
        is_selected = " selected" if value == (selected or "") else ""
        return f'<option value="{esc(value)}"{is_selected}>{esc(label)}</option>'

    state_options = "".join(
        [option("", "Any", state)]
        + [option(value, value, state) for value in _MEMORY_STATES]
    )
    kind_options = "".join(
        [option("", "Any", kind)]
        + [option(value, value, kind) for value in _MEMORY_KINDS]
    )
    about_user_value = "" if about_user is None else ("1" if about_user else "0")
    about_user_options = "".join(
        [
            option("", "Any", about_user_value),
            option("1", "Yes", about_user_value),
            option("0", "No", about_user_value),
        ]
    )
    return f"""
<form method="get" action="{esc(base_path)}">
<label for="state">State</label>
<select id="state" name="state">{state_options}</select>
<label for="kind">Kind</label>
<select id="kind" name="kind">{kind_options}</select>
<label for="about_user">About user</label>
<select id="about_user" name="about_user">{about_user_options}</select>
<button type="submit">Filter</button>
</form>
"""


def _render_memory_detail(space_id: str, detail: MemoryDetail) -> str:
    people_html = (
        "<ul>" + "".join(
            f'<li><a href="/admin/spaces/{esc(space_id)}/people/{esc(p.id)}">'
            f"{esc(p.display_name)}</a></li>"
            for p in detail.people
        ) + "</ul>"
        if detail.people
        else "<p>None linked.</p>"
    )
    relationships_html = (
        "<ul>" + "".join(
            f'<li><a href="/admin/spaces/{esc(space_id)}/relationships/{esc(r.id)}">'
            f"{esc(r.person_a_name)} &harr; {esc(r.person_b_name)}</a>: "
            f"{esc(r.summary or '(no summary)')}</li>"
            for r in detail.relationships
        ) + "</ul>"
        if detail.relationships
        else "<p>None linked.</p>"
    )
    derived_from_html = (
        "<ul>" + "".join(
            f'<li><a href="/admin/spaces/{esc(space_id)}/memories/{esc(d.memory_id)}">'
            f"{esc(d.content[:80])}</a> ({esc(d.role)})</li>"
            for d in detail.derived_from
        ) + "</ul>"
        if detail.derived_from
        else "<p>None.</p>"
    )
    derives_html = (
        "<ul>" + "".join(
            f'<li><a href="/admin/spaces/{esc(space_id)}/memories/{esc(d.memory_id)}">'
            f"{esc(d.content[:80])}</a> ({esc(d.role)})</li>"
            for d in detail.derives
        ) + "</ul>"
        if detail.derives
        else "<p>None.</p>"
    )
    source_messages_html = (
        "<ul>" + "".join(
            f"<li>[{esc(m.occurred_at)}] {esc(m.author)}: {esc(m.content)}</li>"
            for m in detail.source_messages
        ) + "</ul>"
        if detail.source_messages
        else "<p>No source messages (not extracted from a batch).</p>"
    )
    return f"""
<section>
<h2>Content</h2>
<p>{esc(detail.content)}</p>
<ul>
<li>Kind: {esc(detail.kind)}</li>
<li>Basis: {esc(detail.basis)}</li>
<li>Status: {esc(detail.status)}</li>
<li>About user: {esc("yes" if detail.about_user else "no")}</li>
<li>Valid from: {esc(detail.valid_from or "-")}</li>
<li>Valid to: {esc(detail.valid_to or "-")}</li>
<li>Created by: {esc(detail.created_by)}</li>
<li>Created at: {esc(detail.created_at)}</li>
<li>Updated at: {esc(detail.updated_at)}</li>
<li>Supersedes: {esc(detail.supersedes_memory_id or "-")}</li>
<li>Invalidated at: {esc(detail.invalidated_at or "-")}</li>
<li>Invalidation reason: {esc(detail.invalidation_reason or "-")}</li>
</ul>
</section>
<section>
<h2>People</h2>
{people_html}
</section>
<section>
<h2>Relationships</h2>
{relationships_html}
</section>
<section>
<h2>Derived from</h2>
{derived_from_html}
</section>
<section>
<h2>Derives</h2>
{derives_html}
</section>
<section>
<h2>Source messages</h2>
{source_messages_html}
</section>
"""


__all__ = ["register"]
