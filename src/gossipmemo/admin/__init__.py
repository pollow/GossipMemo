"""Read-only, server-rendered admin UI.

Slice 1 delivered auth plumbing and a placeholder landing page. Slice 2
adds the real space/message/memory views under this router. See
docs/adr/0001-admin-ui-server-rendered.md for the two load-bearing
decisions (no template engine/JS, and a separate admin_password/cookie
session instead of reusing the bearer api_key).
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..config import Settings
from ..world import SocialMemoryWorld
from .auth import AdminAuth
from .render import CONTENT_SECURITY_POLICY
from .search import register as register_search_view
from .views import register_admin_views

_STATIC_DIR = Path(__file__).parent / "static"
_CSS_PATH = _STATIC_DIR / "admin.css"


def create_admin_router(settings: Settings, world: SocialMemoryWorld) -> APIRouter | None:
    """Build the /admin router, or None when the admin UI is disabled."""

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

    register_admin_views(router, auth.require_session, world.store)
    register_search_view(router, auth.require_session, world.store)

    return router


__all__ = ["create_admin_router"]
