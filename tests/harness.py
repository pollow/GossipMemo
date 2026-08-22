"""Shared test scaffolding.

Nothing here asserts anything. It exists because a test that touches an
HTTP route cannot get to its first assertion without six steps of
ceremony: a store on a temp path, a `SocialMemoryWorld` wrapped around an
`LlmTransport`, `create_app`, driving the ASGI lifespan (which starts and
stops the background schedulers), an httpx client over `ASGITransport`,
and -- for `/admin` -- a login POST to put a session cookie in the jar.

Each test file used to carry its own copy of all six. They had drifted
only in trivia, but the copies made the suite look larger than the
behavior it covers.

What deliberately did NOT move here: each admin test file's
`_seed_space`. Those are 90-180 lines apiece returning entirely
different key sets, written for one file's assertions. They are fixture
*data*, not scaffolding, and sharing them would couple unrelated files.
`run_admin` takes the seeder as a parameter instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.context_budget import ContextBudget
from gossipmemo.store import SqliteWorldStore
from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
from gossipmemo.world import SocialMemoryWorld

ADMIN_PASSWORD = "correct-horse-battery-staple"
XSS = "<script>alert(1)</script>"


class NoopTransport:
    """An `LlmTransport` double for tests that must never reach the model.

    Deliberately minimal. The admin routes are LLM-free by design, so a
    double with no `prepare`/`complete` turns an accidental model call
    into an `AttributeError` instead of letting it pass silently. Do not
    "helpfully" widen this -- use `SilentTransport` when a test really
    does drive a stage that talks to the transport.
    """

    configured = False

    async def aclose(self):
        return None


class SilentTransport(NoopTransport):
    """An `LlmTransport` double for tests that start the world's pipelines.

    Extraction, continuity, coverage and the owner family all reach the
    transport once `world.start()` runs, so those tests need the full
    protocol. `complete` answers `{}`, which parses against every result
    type (each field has a default and pydantic ignores unknown keys).
    Subclass and override `complete` when a test needs one stage to
    answer something specific -- see `tests/test_turn.py`.
    """

    gate = ProviderGate()
    context_budget = ContextBudget()
    retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)

    def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake",
            messages=list(messages),
            response_format={"type": "json_object"} if structured else None,
        )

    async def complete(self, request: ChatCompletionRequest) -> str:
        return json.dumps({})


def settings(tmp_path: Path, *, admin_password: str = ADMIN_PASSWORD, **overrides) -> Settings:
    """A valid `Settings` for tests.

    `Settings` refuses to construct without LLM base_url/api_key/model --
    that refusal is itself covered by `test_config.py` -- so every test
    has to supply throwaway values even when it never calls a model.
    """
    return Settings(
        database_path=tmp_path / "world.db",
        llm_base_url="http://llm.test/v1",
        llm_api_key="key",
        llm_model="model",
        admin_password=admin_password,
        **overrides,
    )


def asgi_client(app) -> httpx.AsyncClient:
    """An httpx client speaking to `app` in-process, without a socket."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def login(client: httpx.AsyncClient, password: str = ADMIN_PASSWORD) -> None:
    response = await client.post(
        "/admin/login", data={"password": password}, follow_redirects=False
    )
    assert response.status_code == 303


async def run_admin(
    tmp_path: Path,
    scenario,
    *,
    seeder=None,
    seed_spaces=(),
    also_seed=None,
    admin_password: str = ADMIN_PASSWORD,
    authenticate: bool = True,
    settings_overrides: dict | None = None,
    transport=None,
):
    """Build the app, seed, optionally log in, then hand `scenario` the client.

    `scenario(client, fixtures)` runs inside the lifespan context, so
    teardown happens even when an assertion fails. `fixtures` maps each
    seeded space id to whatever that file's `seeder` returned.
    `authenticate=False` is for the tests that check a route redirects to
    the login page. `settings_overrides` reaches `settings()` verbatim --
    e.g. `{"llm_trace_path": some_dir}` for the playground tests.
    `transport` defaults to `NoopTransport()`; the playground replay tests
    pass a double with a working `complete()` instead, since that route is
    the one admin route that reaches the model.
    """
    store = SqliteWorldStore(tmp_path / "world.db")
    world = SocialMemoryWorld(store, transport if transport is not None else NoopTransport())
    app = create_app(
        settings(tmp_path, admin_password=admin_password, **(settings_overrides or {})), world
    )
    fixtures: dict[str, dict] = {}
    async with app.router.lifespan_context(app):
        async with asgi_client(app) as client:
            for space_id in seed_spaces:
                fixtures[space_id] = seeder(store, space_id, f"Space {space_id}")
            if also_seed is not None:
                also_seed(store)
            if authenticate:
                await login(client, admin_password)
            await scenario(client, fixtures)
