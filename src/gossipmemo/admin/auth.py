"""Cookie-based session auth for the admin UI.

Deliberately separate from the bearer `api_key` on `Settings.authorize`:
`api_key` is a machine credential handed to agents, `admin_password` is a
human login. See docs/adr/0001-admin-ui-server-rendered.md.

The session secret is generated once per process (`secrets.token_bytes(32)`
in `create_admin_router`) and never persisted, so a restart invalidates
every outstanding session -- intentional, not a bug.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from .render import add_security_headers, esc, html_response, page

logger = logging.getLogger(__name__)

COOKIE_NAME = "gossipmemo_admin"
SESSION_DURATION = timedelta(hours=12)
LOGIN_PATH = "/admin/login"


class AdminAuth:
    """Owns session signing/verification and the login/logout routes."""

    def __init__(self, secret: bytes, admin_password: str) -> None:
        self._secret = secret
        self._admin_password = admin_password

    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _make_cookie_value(self, *, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        expiry = int((now + SESSION_DURATION).timestamp())
        payload = str(expiry)
        return f"{payload}.{self._sign(payload)}"

    def csrf_token(self) -> str:
        """A process-lifetime CSRF token for the admin UI's one state-changing form.

        Derived from the same session secret and `hmac` construction as
        `_sign`, over a fixed label rather than a per-request nonce -- there
        is exactly one form (the playground replay) and no session-scoped
        state to bind it to beyond the secret itself, which already rotates
        on every process restart.
        """

        return self._sign("csrf")

    def verify_csrf(self, token: str | None) -> bool:
        """True iff `token` matches the current CSRF token, constant-time."""

        if not token:
            return False
        return hmac.compare_digest(token, self.csrf_token())

    def verify(self, cookie_value: str | None) -> bool:
        """True iff `cookie_value` carries a validly-signed, unexpired session.

        Exposed as a plain function (not just via the FastAPI dependency)
        so tests can check signature/expiry handling directly.
        """

        if not cookie_value or "." not in cookie_value:
            return False
        payload, _, signature = cookie_value.rpartition(".")
        expected = self._sign(payload)
        if not hmac.compare_digest(signature, expected):
            return False
        try:
            expiry = int(payload)
        except ValueError:
            return False
        return datetime.now(timezone.utc).timestamp() < expiry

    async def require_session(self, request: Request) -> None:
        """FastAPI dependency: redirect to the login page on any invalid session."""

        if not self.verify(request.cookies.get(COOKIE_NAME)):
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                headers={"Location": LOGIN_PATH},
            )

    def _set_session_cookie(self, response, request: Request) -> None:
        response.set_cookie(
            key=COOKIE_NAME,
            value=self._make_cookie_value(),
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            # No max_age/expires on purpose: this is a browser-session
            # cookie. The absolute expiry embedded in the signed payload is
            # what actually bounds the session's lifetime.
        )

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/login", include_in_schema=False)
        async def login_form(request: Request) -> HTMLResponse:
            return html_response(_login_page())

        @router.post("/login", include_in_schema=False)
        async def login_submit(request: Request) -> HTMLResponse:
            # Parsed by hand rather than via `Request.form()`: that call
            # requires the `python-multipart` dependency even for a plain
            # `application/x-www-form-urlencoded` body, and this UI adds no
            # new dependencies. One text field does not need a form parser.
            body = (await request.body()).decode("utf-8", errors="replace")
            fields = dict(parse_qsl(body, keep_blank_values=True))
            password = fields.get("password", "")
            if not self._admin_password or not secrets.compare_digest(
                password, self._admin_password
            ):
                # Deliberate delay against password-guessing; never log the
                # password itself, only that an attempt failed and from where.
                await asyncio.sleep(1)
                client_host = request.client.host if request.client else "unknown"
                logger.warning(
                    "admin_login_failed", extra={"client_ip": client_host}
                )
                return html_response(
                    _login_page(error="Incorrect password."),
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )
            response = add_security_headers(
                RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
            )
            self._set_session_cookie(response, request)
            return response

        @router.post("/logout", include_in_schema=False)
        async def logout() -> RedirectResponse:
            response = add_security_headers(
                RedirectResponse(url=LOGIN_PATH, status_code=status.HTTP_303_SEE_OTHER)
            )
            response.delete_cookie(COOKIE_NAME)
            return response

        return router


def _login_page(error: str | None = None) -> str:
    error_html = f'<p class="error">{esc(error)}</p>' if error else ""
    body = f"""
{error_html}
<form method="post" action="{esc(LOGIN_PATH)}">
<label for="password">Password</label>
<input type="password" id="password" name="password" autocomplete="current-password" required>
<button type="submit">Log in</button>
</form>
"""
    return page(title="GossipMemo Admin Login", body=body)


__all__ = ["AdminAuth", "COOKIE_NAME", "LOGIN_PATH", "SESSION_DURATION"]
