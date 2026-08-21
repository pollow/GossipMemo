# 0001: Admin UI is hand-written server-rendered HTML with its own cookie session

## Context

GossipMemo needs a read-only admin UI for inspecting a space's People,
Memories, Relationships, and provenance -- data that today is only reachable
through `curl`ing the JSON API. Two independent decisions shape it.

### (a) Rendering approach

Options considered: a template engine (Jinja2), a frontend app (a small
React/Vue SPA calling the JSON API), or hand-written HTML strings with no
JavaScript.

### (b) Auth

The app already has one credential, `Settings.api_key`, checked as a bearer
token on every `/v1/...` route via the `authorize` dependency. The admin UI
could reuse it.

## Decision

**(a)** Hand-written server-rendered HTML via f-strings, with zero
JavaScript and zero new dependencies. `admin/render.py` provides a page
skeleton, an `esc()` escaping helper used on every interpolated value, and
reusable table/pagination/breadcrumb components; views are plain functions
that call them.

**(b)** A separate `admin_password` setting with signed, cookie-based
sessions (`admin/auth.py`), entirely independent of `api_key`. Login
compares the submitted password against `admin_password` with
`secrets.compare_digest`; on success a cookie carries an absolute 12-hour
expiry plus an HMAC-SHA256 signature over that expiry, keyed by a session
secret generated once per process. There is no sliding renewal, no
persistence of the secret, and no crossover with `authorize`: admin routes
never accept a bearer token, and `/v1/...` routes never accept the admin
cookie.

## Consequences

**(a)** This caps what the admin UI can ever be: no client-side
interactivity, no partial-page updates, no rich widgets -- every view is a
full page load, and CSP is `default-src 'none'; style-src 'self';
form-action 'self'`, which would break the first `<script>` anyone added
without also revisiting this ADR. That's the point: a read-only inspector
for a local-first, single-user server doesn't need Jinja2's template
inheritance or an SPA's client state management, and both would add a
dependency (or a build step) to audit and keep patched for a UI whose
entire job is to print rows from SQLite. If a later slice genuinely needs
client-side behavior (e.g. a live-updating view), that is a deliberate
scope change and should get its own ADR rather than sneaking in through
"just one script tag."

**(b)** `api_key` is a machine credential: it's handed to agents and other
programs, typically kept in an env var or config file. Nothing about it
signals "safe to type into a browser," yet a browser login checking it
would make every place `api_key` is copied around (shell history, agent
config, logs an agent might produce) a place capable of controlling the
admin UI too. Keeping `admin_password` and its cookie session separate
means rotating one credential never affects the other, and the admin UI
can be disabled outright (empty `admin_password`) without touching API
access, or vice versa. The cost is a second credential to provision and a
small amount of duplicated "check a secret, log failures, sleep on
mismatch" logic that doesn't share code with `authorize` -- acceptable
given how different the two threat models are (a bearer token replayed by
a script vs. a password typed into a browser and worth guarding against
guessing).
