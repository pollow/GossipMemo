"""Read-only admin views: whitelisted raw operational tables.

`schema_migrations`, `extraction_batches`, and `embeddings` are the only
operational tables with no domain view of their own; a generic paginated
row dump is the right shape for them. `ADMIN_RAW_TABLES` is the strict
whitelist -- any other name is a 404, checked here before the store is
ever called. The table name is never interpolated into SQL:
`_AdminReadMixin.admin_list_raw_table` maps each whitelist entry to a
hardcoded query.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ...store._admin import ADMIN_RAW_TABLES
from ...store.sqlite import SqliteWorldStore
from ..render import html_response, page, table_component
from ._common import clamp_limit, clamp_offset


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/tables", include_in_schema=False)
    async def tables_index(_: None = Depends(require_session)) -> HTMLResponse:
        table_html = table_component(
            headers=["Table"],
            rows=[[name] for name in ADMIN_RAW_TABLES],
            row_hrefs=[f"/admin/tables/{name}" for name in ADMIN_RAW_TABLES],
            offset=0,
            limit=max(len(ADMIN_RAW_TABLES), 1),
            total=len(ADMIN_RAW_TABLES),
            base_path="/admin/tables",
        )
        return html_response(
            page(
                title="Raw tables",
                breadcrumbs=[("Admin", "/admin"), ("Tables", "/admin/tables")],
                body=table_html,
            )
        )

    @router.get("/tables/{name}", include_in_schema=False)
    async def table_view(
        name: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        breadcrumbs = [
            ("Admin", "/admin"),
            ("Tables", "/admin/tables"),
            (name, f"/admin/tables/{name}"),
        ]
        if name not in ADMIN_RAW_TABLES:
            return html_response(
                page(
                    title="Table not found",
                    breadcrumbs=breadcrumbs,
                    body="<p>No such table.</p>",
                ),
                status_code=404,
            )
        query = request.query_params
        offset = clamp_offset(query.get("offset"))
        limit = clamp_limit(query.get("limit"))
        total = store.admin_count_raw_table(name)
        table_page = store.admin_list_raw_table(name, offset, limit)
        table_html = table_component(
            headers=table_page.headers,
            rows=table_page.rows,
            offset=offset,
            limit=limit,
            total=total,
            base_path=f"/admin/tables/{name}",
        )
        return html_response(
            page(title=f"Table: {name}", breadcrumbs=breadcrumbs, body=table_html)
        )


__all__ = ["register"]
