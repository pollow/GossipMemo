"""Shared helpers for the admin view modules.

Split out of the original single `views.py` once slice 3 added enough
routes to push it past a readable size. Every entity module
(`spaces.py`, `people.py`, `relationships.py`, `goals.py`,
`hypotheses.py`, `coverage.py`) imports from here rather than from each
other.
"""

from __future__ import annotations

from ...store.sqlite import SqliteWorldStore
from ..render import DEFAULT_PAGE_SIZE, html_response, page


def clamp_offset(raw: str | None) -> int:
    try:
        value = int(raw) if raw is not None else 0
    except ValueError:
        return 0
    return max(0, value)


def clamp_limit(raw: str | None) -> int:
    try:
        value = int(raw) if raw is not None else DEFAULT_PAGE_SIZE
    except ValueError:
        return DEFAULT_PAGE_SIZE
    return min(max(1, value), DEFAULT_PAGE_SIZE)


def clean_choice(raw: str | None, allowed: tuple[str, ...]) -> str | None:
    """A filter value only survives if it is one of `allowed`; anything
    else (including an empty "no filter" choice) is dropped rather than
    ever reaching SQL unbound."""

    if raw and raw in allowed:
        return raw
    return None


def clean_bool(raw: str | None) -> bool | None:
    if raw == "1":
        return True
    if raw == "0":
        return False
    return None


def space_breadcrumbs(space_id: str, space_name: str) -> list[tuple[str, str]]:
    return [
        ("Admin", "/admin"),
        ("Spaces", "/admin/spaces"),
        (space_name, f"/admin/spaces/{space_id}"),
    ]


def require_space(store: SqliteWorldStore, space_id: str):
    """Look up a space's overview, or an already-rendered 404 page.

    Every entity view under `/admin/spaces/{space_id}/...` needs the same
    "no such space" guard before it can render breadcrumbs; this keeps that
    check (and its 404 body) in one place. Callers do:

        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
    """

    overview = store.admin_space_overview(space_id)
    if overview is None:
        return html_response(
            page(
                title="Space not found",
                breadcrumbs=[("Admin", "/admin"), ("Spaces", "/admin/spaces")],
                body="<p>No such space.</p>",
            ),
            status_code=404,
        )
    return overview


__all__ = [
    "clamp_limit",
    "clamp_offset",
    "clean_bool",
    "clean_choice",
    "require_space",
    "space_breadcrumbs",
]
