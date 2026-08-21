"""Hand-written HTML rendering for the read-only admin UI.

No template engine, no JavaScript. Every interpolated value must go
through `esc()`. Kept deliberately small: later slices add views, this
module only provides the page skeleton and the components every view
will reuse (table, pagination, breadcrumbs).
"""

from __future__ import annotations

import html
import json

from fastapi.responses import HTMLResponse

#: Locked down: no scripts, no external resources, no framing. The
#: admin UI has no need for any of that, and it must not gain the
#: ability to load one by accident.
CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'self'; form-action 'self'"

DEFAULT_PAGE_SIZE = 50


def esc(value: object) -> str:
    """Escape a value for safe interpolation into HTML text or attributes."""

    return html.escape(str(value), quote=True)


def json_block(raw: object) -> str:
    """Render a stored JSON blob as an indented `<pre>` block.

    Cards and other JSON columns are stored compact, so a raw dump is one
    unreadable line. This re-indents it server-side -- the admin UI has no
    JavaScript, so the formatting has to happen here. Anything that is not
    parseable JSON (or is not a string at all) is shown verbatim rather
    than swallowed: the point of this view is to see what is actually in
    the database.
    """

    text = "" if raw is None else str(raw)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        pretty = text
    else:
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True)
    return f'<pre>{esc(pretty)}</pre>'


def add_security_headers(response):
    """Stamp the CSP and framing headers every admin response must carry."""

    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Frame-Options"] = "DENY"
    return response


def html_response(body: str, status_code: int = 200) -> HTMLResponse:
    """Wrap a fully-rendered HTML document with the required security headers."""

    return add_security_headers(HTMLResponse(content=body, status_code=status_code))


def breadcrumbs_component(items: list[tuple[str, str]]) -> str:
    """Render a breadcrumb trail. `items` is `[(label, href), ...]`; the last
    item is rendered as plain text (it is the current page)."""

    if not items:
        return ""
    parts: list[str] = []
    last_index = len(items) - 1
    for index, (label, href) in enumerate(items):
        if index == last_index:
            parts.append(f"<span>{esc(label)}</span>")
        else:
            parts.append(f'<a href="{esc(href)}">{esc(label)}</a>')
    return '<nav aria-label="breadcrumb">' + " &raquo; ".join(parts) + "</nav>"


def table_component(
    *,
    headers: list[str],
    rows: list[list[object]],
    offset: int,
    limit: int,
    total: int,
    base_path: str,
    extra_params: dict[str, str] | None = None,
    row_hrefs: list[str | None] | None = None,
) -> str:
    """Render a table plus an offset/limit prev/next pagination footer.

    `extra_params` carries filters (e.g. a search query) that must survive
    into the prev/next links alongside the offset/limit pair.

    `row_hrefs`, when given, is one URL (or None) per row in `rows`: the
    row's first cell is wrapped in a link to it, so a list page can link
    each row into its detail page without any view writing raw HTML of its
    own. The href itself is escaped like any other interpolated value.
    """

    extra_params = extra_params or {}
    head_html = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body_rows: list[str] = []
    for index, row in enumerate(rows):
        href = row_hrefs[index] if row_hrefs else None
        cells_html: list[str] = []
        for column, cell in enumerate(row):
            if column == 0 and href is not None:
                cells_html.append(f'<td><a href="{esc(href)}">{esc(cell)}</a></td>')
            else:
                cells_html.append(f"<td>{esc(cell)}</td>")
        body_rows.append("<tr>" + "".join(cells_html) + "</tr>")
    body_html = "".join(body_rows)
    if not rows:
        body_html = f'<tr><td colspan="{len(headers)}">No results.</td></tr>'

    def _link(new_offset: int) -> str:
        params = dict(extra_params)
        params["offset"] = str(new_offset)
        params["limit"] = str(limit)
        query_string = "&".join(f"{esc(key)}={esc(value)}" for key, value in params.items())
        return f"{esc(base_path)}?{query_string}"

    prev_html = (
        f'<a href="{_link(max(0, offset - limit))}">Previous</a>'
        if offset > 0
        else "<span>Previous</span>"
    )
    next_offset = offset + limit
    next_html = (
        f'<a href="{_link(next_offset)}">Next</a>'
        if next_offset < total
        else "<span>Next</span>"
    )
    range_start = 0 if total == 0 else offset + 1
    range_end = min(offset + limit, total)
    return (
        f"<table><thead><tr>{head_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table>"
        f'<p class="pagination">{prev_html} | {next_html} '
        f"&mdash; {range_start}-{range_end} of {total}</p>"
    )


def page(*, title: str, body: str, breadcrumbs: list[tuple[str, str]] | None = None) -> str:
    """Wrap `body` (already-rendered HTML) in the admin page skeleton."""

    crumbs_html = breadcrumbs_component(breadcrumbs or [])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<link rel="stylesheet" href="/admin/static/admin.css">
</head>
<body>
<header><h1>{esc(title)}</h1></header>
<main>
{crumbs_html}
{body}
</main>
</body>
</html>
"""


__all__ = [
    "CONTENT_SECURITY_POLICY",
    "DEFAULT_PAGE_SIZE",
    "add_security_headers",
    "breadcrumbs_component",
    "esc",
    "html_response",
    "json_block",
    "page",
    "table_component",
]
