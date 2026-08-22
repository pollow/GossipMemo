"""Read-only admin views: coverage roots and one root's entries.

Coverage is two levels: the root list, then a drill-down into one root's
entries. A root-level overview entry is just the entry whose `path` is
empty -- same type as any other entry, only a different granularity, so it
is rendered inline in the same list rather than pulled out as a special
row (see glossary.md).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ...store.sqlite import SqliteWorldStore
from ..render import esc, html_response, page, table_component
from ._common import clamp_limit, clamp_offset, require_space, space_breadcrumbs


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/spaces/{space_id}/coverage", include_in_schema=False)
    async def coverage_roots_view(
        space_id: str, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        rows = store.admin_list_coverage_roots(space_id)
        base_path = f"/admin/spaces/{space_id}/coverage"
        table_html = table_component(
            headers=["Root", "Entries", "Revision", "Source watermark"],
            rows=[
                [row.root, row.entry_count, row.revision, row.source_watermark or "-"]
                for row in rows
            ],
            column_classes=["nowrap", "num nowrap", "num nowrap", "nowrap mono"],
            row_hrefs=[f"{base_path}/{row.root}" for row in rows],
            offset=0,
            limit=max(len(rows), 1),
            total=len(rows),
            base_path=base_path,
        )
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [("Coverage", base_path)]
        return html_response(
            page(title=f"Coverage: {overview.name}", breadcrumbs=breadcrumbs, body=table_html)
        )

    @router.get("/spaces/{space_id}/coverage/{root}", include_in_schema=False)
    async def coverage_entries_view(
        space_id: str, root: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        roots_base_path = f"/admin/spaces/{space_id}/coverage"
        coverage_root = store.admin_get_coverage_root(space_id, root)
        if coverage_root is None:
            breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
                ("Coverage", roots_base_path),
                (root, f"{roots_base_path}/{root}"),
            ]
            return html_response(
                page(title="Coverage root not found", breadcrumbs=breadcrumbs,
                     body="<p>No such coverage root.</p>"),
                status_code=404,
            )
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        total = store.admin_count_coverage_entries(space_id, root)
        rows = store.admin_list_coverage_entries(space_id, root, offset, limit)
        base_path = f"{roots_base_path}/{root}"
        table_html = table_component(
            headers=["Path", "Content", "Updated at"],
            rows=[
                [
                    row.path or "(root overview)",
                    (row.content[:400] + "...") if len(row.content) > 400 else row.content,
                    row.updated_at,
                ]
                for row in rows
            ],
            column_classes=["nowrap mono", "wrap", "nowrap mono"],
            offset=offset,
            limit=limit,
            total=total,
            base_path=base_path,
        )
        summary = (
            f"<p>Revision: {esc(coverage_root.revision)} &mdash; "
            f"Source watermark: {esc(coverage_root.source_watermark or 'none')}</p>"
        )
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [
            ("Coverage", roots_base_path),
            (root, base_path),
        ]
        return html_response(
            page(
                title=f"Coverage root {root}: {overview.name}",
                breadcrumbs=breadcrumbs,
                body=summary + table_html,
            )
        )


__all__ = ["register"]
