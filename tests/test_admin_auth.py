from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from gossipmemo.admin.auth import COOKIE_NAME, AdminAuth
from gossipmemo.app import create_app
from gossipmemo.config import ConfigurationError, Settings
from gossipmemo.store import SqliteWorldStore
from gossipmemo.world import SocialMemoryWorld


class _NoopModel:
    """Minimal `LlmTransport` double -- the admin routes never call the
    model, but `SocialMemoryWorld` needs one to construct."""

    configured = False

    async def aclose(self):
        return None


def _settings(tmp_path: Path, *, admin_password: str = "") -> Settings:
    return Settings(
        database_path=tmp_path / "world.db",
        llm_base_url="http://llm.test/v1",
        llm_api_key="key",
        llm_model="model",
        admin_password=admin_password,
    )


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _run(tmp_path: Path, admin_password: str, scenario):
    store = SqliteWorldStore(tmp_path / "world.db")
    world = SocialMemoryWorld(store, _NoopModel())
    app = create_app(_settings(tmp_path, admin_password=admin_password), world)
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            await scenario(client)


def test_admin_password_shorter_than_12_chars_raises(tmp_path: Path):
    with pytest.raises(ConfigurationError):
        _settings(tmp_path, admin_password="short")


def test_admin_disabled_returns_404_and_registers_no_routes(tmp_path: Path):
    async def scenario():
        store = SqliteWorldStore(tmp_path / "world.db")
        world = SocialMemoryWorld(store, _NoopModel())
        app = create_app(_settings(tmp_path, admin_password=""), world)
        async with app.router.lifespan_context(app):
            async with _client(app) as client:
                response = await client.get("/admin")
                assert response.status_code == 404

        paths = {route.path for route in app.router.routes}
        assert not any(path.startswith("/admin") for path in paths)

    asyncio.run(scenario())


def test_wrong_password_rerenders_form_without_cookie(tmp_path: Path):
    async def scenario(client):
        response = await client.post(
            "/admin/login", data={"password": "totally-wrong-password"}
        )
        assert response.status_code == 401
        assert "Incorrect password" in response.text
        assert COOKIE_NAME not in response.cookies

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))


def test_correct_password_sets_httponly_samesite_strict_cookie_without_secure_over_http(
    tmp_path: Path,
):
    async def scenario(client):
        response = await client.post(
            "/admin/login",
            data={"password": "correct-horse-battery-staple"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        set_cookie = response.headers.get("set-cookie", "")
        assert COOKIE_NAME in response.cookies
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie.lower().replace("samesite", "SameSite") or (
            "samesite=strict" in set_cookie.lower()
        )
        assert "secure" not in set_cookie.lower()

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))


def test_landing_page_reachable_after_login(tmp_path: Path):
    async def scenario(client):
        login = await client.post(
            "/admin/login",
            data={"password": "correct-horse-battery-staple"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        landing = await client.get("/admin")
        assert landing.status_code == 200
        assert "No results" in landing.text
        # CSP is set centrally in `admin/render.py`, so one rendered page
        # standing in for all of them is enough.
        assert "Content-Security-Policy" in landing.headers
        assert landing.headers["X-Frame-Options"] == "DENY"

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))


def test_missing_session_redirects_to_login(tmp_path: Path):
    async def scenario(client):
        response = await client.get("/admin", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))


def test_tampered_cookie_signature_is_rejected():
    auth = AdminAuth(secret=b"secret-a" * 4, admin_password="correct-horse-battery")
    genuine = auth._make_cookie_value()
    payload, _, signature = genuine.rpartition(".")
    tampered = f"{payload}.{'0' * len(signature)}"
    assert auth.verify(genuine) is True
    assert auth.verify(tampered) is False


def test_expired_payload_is_rejected():
    auth = AdminAuth(secret=b"secret-b" * 4, admin_password="correct-horse-battery")
    expired = auth._make_cookie_value(now=datetime.now(timezone.utc) - timedelta(hours=13))
    assert auth.verify(expired) is False


def test_logout_clears_cookie(tmp_path: Path):
    async def scenario(client):
        await client.post(
            "/admin/login",
            data={"password": "correct-horse-battery-staple"},
            follow_redirects=False,
        )
        assert client.cookies.get(COOKIE_NAME)
        logout = await client.post("/admin/logout", follow_redirects=False)
        assert logout.status_code == 303
        assert logout.headers["location"] == "/admin/login"
        set_cookie = logout.headers.get("set-cookie", "")
        assert COOKIE_NAME in set_cookie

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))


def test_static_css_served_without_session_as_text_css(tmp_path: Path):
    async def scenario(client):
        response = await client.get("/admin/static/admin.css")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/css")

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))


@pytest.mark.parametrize(
    "make_request",
    [
        lambda client: client.get("/admin/login"),
        lambda client: client.get("/admin/static/admin.css"),
    ],
)
def test_admin_responses_carry_csp_header(tmp_path: Path, make_request):
    async def scenario(client):
        response = await make_request(client)
        assert "Content-Security-Policy" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

    asyncio.run(_run(tmp_path, "correct-horse-battery-staple", scenario))
