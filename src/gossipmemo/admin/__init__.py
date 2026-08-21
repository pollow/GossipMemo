"""Read-only, server-rendered admin UI.

Slice 1 delivers only auth plumbing and a placeholder landing page; later
slices add real views under this router. See
docs/adr/0001-admin-ui-server-rendered.md for the two load-bearing
decisions (no template engine/JS, and a separate admin_password/cookie
session instead of reusing the bearer api_key).
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse

from ..config import Settings
from ..world import SocialMemoryWorld
from .auth import AdminAuth
from .render import CONTENT_SECURITY_POLICY, html_response, page

_STATIC_DIR = Path(__file__).parent / "static"
_CSS_PATH = _STATIC_DIR / "simple.css"


def create_admin_router(settings: Settings, world: SocialMemoryWorld) -> APIRouter | None:
    """Build the /admin router, or None when the admin UI is disabled.

    `world` is accepted (and will be used by later slices' views) even
    though slice 1's placeholder landing page does not touch it.
    """

    if not settings.admin_password:
        return None

    # Process-local, generated once per router (i.e. once per process
    # start): a restart invalidates every outstanding session. Intentional.
    secret = secrets.token_bytes(32)
    auth = AdminAuth(secret=secret, admin_password=settings.admin_password)

    router = APIRouter(prefix="/admin")

    @router.get("/static/admin.css", include_in_schema=False)
    async def admin_css() -> FileResponse:
        return FileResponse(
            _CSS_PATH,
            media_type="text/css",
            headers={
                "Cache-Control": "public, max-age=86400, immutable",
                "Content-Security-Policy": CONTENT_SECURITY_POLICY,
                "X-Frame-Options": "DENY",
            },
        )

    router.include_router(auth.router())

    @router.get("", include_in_schema=False)
    async def landing(_: None = Depends(auth.require_session)) -> HTMLResponse:
        body = (
            "<p>The admin UI is under construction. "
            "Later slices will add People, Memories, and Relationships views here.</p>"
            f'<form method="post" action="/admin/logout"><button type="submit">Log out</button></form>'
        )
        return html_response(
            page(title="GossipMemo Admin", breadcrumbs=[("Admin", "/admin")], body=body)
        )

    return router


__all__ = ["create_admin_router"]
