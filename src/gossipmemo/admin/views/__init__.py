"""Read-only, server-rendered admin views, split by entity.

Slice 2 shipped spaces/messages/memories as a single `views.py`. Slice 3
added people, relationships, learning goals, hypotheses, and coverage,
which pushed the module past a readable size, so it became this package:
one module per entity, each exposing a `register(router, require_session,
store)` function that attaches its own routes. `register_admin_views`
below is the only symbol `admin/__init__.py` imports.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter

from ...store.sqlite import SqliteWorldStore
from ...transport import LlmTransport
from . import coverage, goals, hypotheses, people, playground, relationships, spaces, tables


def register_admin_views(
    router: APIRouter,
    require_session,
    store: SqliteWorldStore,
    trace_dir: Path | None = None,
    *,
    admin_playground_enabled: bool = False,
    model: LlmTransport | None = None,
    csrf_token: Callable[[], str] | None = None,
    verify_csrf: Callable[[str | None], bool] | None = None,
) -> None:
    """Attach every read-only admin view to `router`.

    `require_session` is `AdminAuth.require_session`, passed in rather than
    imported so no view module ever constructs its own `AdminAuth`.
    `trace_dir` is `Settings.llm_trace_path`; only `playground` uses it.
    `admin_playground_enabled`, `model`, `csrf_token`, and `verify_csrf`
    gate and drive `playground`'s replay form and POST route -- every other
    view ignores them.
    """

    spaces.register(router, require_session, store)
    people.register(router, require_session, store)
    relationships.register(router, require_session, store)
    goals.register(router, require_session, store)
    hypotheses.register(router, require_session, store)
    coverage.register(router, require_session, store)
    tables.register(router, require_session, store)
    playground.register(
        router,
        require_session,
        trace_dir,
        enabled=admin_playground_enabled,
        model=model,
        csrf_token=csrf_token,
        verify_csrf=verify_csrf,
    )


__all__ = ["register_admin_views"]
