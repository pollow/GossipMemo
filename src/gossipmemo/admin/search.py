"""Read-only admin view: keyword search across one space.

Scans seven kinds -- memories, messages, people (name and aliases),
learning goals, hypotheses, coverage entries, continuities -- each with a
plain `LIKE` query (see `store/_admin.py::admin_search`; this slice
deliberately does not add an FTS index for messages). Every result links
into the detail or list page that already exists from earlier slices;
kinds without a per-row detail page (messages, learning goals,
hypotheses, coverage entries, continuities) link to their existing list
or overview page instead.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..store._admin import SearchGroup, SearchResults
from ..store.sqlite import SqliteWorldStore
from .render import esc, html_response, page
from .views._common import require_space, space_breadcrumbs

_SNIPPET_WIDTH = 160


def register(router: APIRouter, require_session, store: SqliteWorldStore) -> None:
    @router.get("/spaces/{space_id}/search", include_in_schema=False)
    async def search_view(
        space_id: str, request: Request, _: None = Depends(require_session)
    ) -> HTMLResponse:
        overview = require_space(store, space_id)
        if isinstance(overview, HTMLResponse):
            return overview
        base_path = f"/admin/spaces/{space_id}/search"
        raw_query = request.query_params.get("q", "")
        keyword = raw_query.strip()
        breadcrumbs = space_breadcrumbs(space_id, overview.name) + [("Search", base_path)]

        form_html = _search_form(base_path, raw_query)
        if not keyword:
            body = form_html + "<p>Enter a keyword to search this space.</p>"
            return html_response(
                page(title=f"Search: {overview.name}", breadcrumbs=breadcrumbs, body=body)
            )

        results = store.admin_search(space_id, keyword)
        body = form_html + _render_results(space_id, results, keyword)
        return html_response(
            page(title=f"Search: {overview.name}", breadcrumbs=breadcrumbs, body=body)
        )


def _search_form(base_path: str, raw_query: str) -> str:
    return f"""
<form method="get" action="{esc(base_path)}">
<label for="q">Search</label>
<input type="text" id="q" name="q" value="{esc(raw_query)}">
<button type="submit">Search</button>
</form>
"""


def _snippet(text: str, keyword: str, width: int = _SNIPPET_WIDTH) -> str:
    """A short, simple excerpt around the first match. No highlighting
    markup: that would reopen an escaping hole for no real benefit here."""

    if len(text) <= width:
        return text
    index = text.lower().find(keyword.lower())
    if index == -1:
        start, end = 0, width
    else:
        half = width // 2
        start = max(0, index - half)
        end = min(len(text), start + width)
        start = max(0, end - width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _group_section(*, title: str, group: SearchGroup, render_item) -> str:
    count = len(group.hits)
    heading = f"<h2>{esc(title)} ({count})</h2>"
    if not group.hits:
        return heading + "<p>No matches.</p>"
    items = "".join(render_item(hit) for hit in group.hits)
    truncation_note = (
        f'<p class="truncated">Showing the first {count} matches for '
        f"{esc(title.lower())} &mdash; refine your keyword for more.</p>"
        if group.truncated
        else ""
    )
    return heading + f"<ul>{items}</ul>" + truncation_note


def _render_results(space_id: str, results: SearchResults, keyword: str) -> str:
    memories_path = f"/admin/spaces/{space_id}/memories"
    messages_path = f"/admin/spaces/{space_id}/messages"
    people_path = f"/admin/spaces/{space_id}/people"
    goals_path = f"/admin/spaces/{space_id}/goals"
    hypotheses_path = f"/admin/spaces/{space_id}/hypotheses"
    coverage_path = f"/admin/spaces/{space_id}/coverage"
    space_path = f"/admin/spaces/{space_id}"

    def memory_item(hit) -> str:
        href = f"{memories_path}/{hit.id}"
        return f'<li><a href="{esc(href)}">{esc(_snippet(hit.content, keyword))}</a></li>'

    def message_item(hit) -> str:
        snippet = _snippet(hit.content, keyword)
        return (
            f'<li><a href="{esc(messages_path)}">'
            f"[{esc(hit.occurred_at)}] {esc(snippet)}</a></li>"
        )

    def person_item(hit) -> str:
        href = f"{people_path}/{hit.id}"
        return f'<li><a href="{esc(href)}">{esc(hit.display_name)}</a></li>'

    def goal_item(hit) -> str:
        return f'<li><a href="{esc(goals_path)}">{esc(_snippet(hit.prompt, keyword))}</a></li>'

    def hypothesis_item(hit) -> str:
        return (
            f'<li><a href="{esc(hypotheses_path)}">'
            f"{esc(_snippet(hit.content, keyword))}</a></li>"
        )

    def coverage_item(hit) -> str:
        href = f"{coverage_path}/{hit.root}"
        location = hit.path or "(root overview)"
        return (
            f'<li><a href="{esc(href)}">{esc(hit.root)} / {esc(location)}: '
            f"{esc(_snippet(hit.content, keyword))}</a></li>"
        )

    def continuity_item(hit) -> str:
        return f'<li><a href="{esc(space_path)}">{esc(_snippet(hit.text, keyword))}</a></li>'

    sections = [
        _group_section(title="Memories", group=results.memories, render_item=memory_item),
        _group_section(title="Messages", group=results.messages, render_item=message_item),
        _group_section(title="People", group=results.people, render_item=person_item),
        _group_section(
            title="Learning goals", group=results.learning_goals, render_item=goal_item
        ),
        _group_section(
            title="Hypotheses", group=results.hypotheses, render_item=hypothesis_item
        ),
        _group_section(
            title="Coverage entries", group=results.coverage_entries,
            render_item=coverage_item,
        ),
        _group_section(
            title="Continuity", group=results.continuities, render_item=continuity_item
        ),
    ]
    return "".join(f"<section>{section}</section>" for section in sections)


__all__ = ["register"]
