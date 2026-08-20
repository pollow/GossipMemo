# GossipMemo

GossipMemo is a local-first social memory server for agents, built as a
supplement for self-hosted personal assistants. It models the people in your
life and your relationships with them, and it keeps track of *how* it knows
each thing — what you stated first-hand, what someone else reported to you, and
what the system merely inferred.

That provenance distinction is the point. A memory layer that flattens "Bob is
leaving the company" and "Alice thinks Bob may be leaving" into one fact will
eventually make an agent assert the second as if it were the first. GossipMemo
keeps the basis attached to every record, so hedged and second-hand claims stay
hedged and second-hand all the way to the prompt.

It runs as one FastAPI process over a SQLite volume, with a single-permit
priority gate in front of all outbound LLM provider requests, and ships with
a Hermes memory-provider plugin.

## Run with Docker

```bash
cp .env.example .env
# Set GOSSIPMEMO_LLM_BASE_URL, GOSSIPMEMO_LLM_MODEL, and
# GOSSIPMEMO_LLM_API_KEY in .env.
docker compose up --build
```

The server listens on `http://localhost:8765`. Its health endpoint is
`GET /healthz`; interactive OpenAPI documentation is available at `/docs`.
Set `GOSSIPMEMO_DATA_DIR` in `.env` to bind mount a host directory for db.

## Run from source

```bash
uv sync --extra server --extra dev
cp .env.example .env
# Fill all three required GOSSIPMEMO_LLM_* values, then:
uv run --env-file .env gossipmemo serve
```

## Config

Server startup is strict: `GOSSIPMEMO_LLM_BASE_URL`,
`GOSSIPMEMO_LLM_MODEL`, and `GOSSIPMEMO_LLM_API_KEY` must all be non-empty.
There is no default provider URL or no-LLM query fallback. Configuration is
read once at process initialization and passed to the server modules as one
immutable `Settings` value.

`GOSSIPMEMO_USER_NAME` names the fixed current user in LLM prompts (default:
`CurrentUser`).

Extraction batches wait for 6 messages by default. A partial batch is flushed
when its oldest message has waited 30 minutes. These values can be changed with
`GOSSIPMEMO_EXTRACTION_BATCH_SIZE` and
`GOSSIPMEMO_EXTRACTION_BATCH_TIMEOUT_SECONDS`.

Transient LLM failures (`408`, `429`, `5xx`, and transport errors) are retried
with jittered exponential backoff. Configure the retry count, initial delay,
and delay cap with `GOSSIPMEMO_LLM_MAX_RETRIES`,
`GOSSIPMEMO_LLM_RETRY_BASE_SECONDS`, and
`GOSSIPMEMO_LLM_RETRY_MAX_SECONDS`; defaults are 5, 1 second, and 30 seconds.

LLM requests use a conservative context budget: the default 64k-token window
reserves 8k output tokens and 4k safety tokens. Configure these with
`GOSSIPMEMO_LLM_CONTEXT_WINDOW_TOKENS`, `GOSSIPMEMO_LLM_OUTPUT_RESERVE_TOKENS`,
and `GOSSIPMEMO_LLM_CONTEXT_SAFETY_TOKENS`. Requests that exceed the usable
input budget fail before any provider HTTP call. Reasoning catch-up runs through
the internal ordered pipeline.

Profile induction runs once per day at local midnight. Startup performs one
stale-profile catch-up before waiting for the next induction run.

Application logs go to stderr in structured JSON at `INFO` by default. Set
`GOSSIPMEMO_LOG_LEVEL` to `DEBUG`, `WARNING`, or another standard level, and
`GOSSIPMEMO_LOG_FORMAT=text` for local development. HTTP requests receive (or
preserve) an `X-Request-ID`; logs include only request metadata, identifiers,
counts, status, and durations—not message bodies, bearer tokens, or LLM API
keys.

To audit prompt construction, set `GOSSIPMEMO_LLM_TRACE_PATH` to a JSONL file.
Every provider request is then appended verbatim together with its completion,
the reasoner label that issued it (`audit-coverage`, `plan-learning-goals`, and
so on), and the estimated token count. The trace holds full message bodies by
design, so it is a local debugging tool rather than something to leave on: it is
off unless the variable is set, and a write failure never interrupts reasoning.

## Import existing chats

The CLI imports a JSON array, a `{ "messages": [...] }` export, or JSONL. Run
it with repeated `--chat` options; `--user-md` explicitly replaces the initial
UserModel card with the Markdown contents. Imports are idempotent, drain all
message extraction before exiting, and print a compact JSON summary:

```bash
uv run --env-file .env gossipmemo import --space personal \
  --chat export.jsonl --user-md USER.md
```

Each chat record needs `author` (or `role`), `content`, and a timezone-aware
`occurred_at` sender timestamp. Extraction options are optional. Records from
multiple input files are persisted in sender-time order.

## Smoke test a running server

After starting the server, run:

```bash
uv run python scripts/smoke_test.py
```

The script uses a separate `smoke-test` Space by default. It exercises health,
automatic ingest/query, and the manual memory, supersede, and retract flow.
Useful options:

```bash
uv run python scripts/smoke_test.py --space personal
uv run python scripts/smoke_test.py --skip-ingest
```

`GOSSIPMEMO_BASE_URL`, `GOSSIPMEMO_API_KEY`, and `GOSSIPMEMO_SPACE_ID` are also
accepted from the environment.

## Evaluate real-conversation fixtures

`scripts/eval_real_conversations.py` runs text fixtures through the complete
extraction, projection-induction, and query-synthesis workflow using isolated
SQLite databases. It never writes to the configured production database. The
fixture format and built-in cases are documented by `--help`.

```bash
UV_CACHE_DIR=/tmp/gossipmemo-uv-cache \
  uv run --env-file .env python scripts/eval_real_conversations.py \
  gossipmemo_test_data.txt --case fixture-01 \
  --report /tmp/gossipmemo-real-eval-report.json
```

Omit `--case` to run all independent and cross-session cases. Repeat `--case`
to select several. Add `--database-dir /private/path` to retain the case
databases for inspection; the directory, databases, and JSON report are given
private filesystem permissions because fixtures may contain sensitive text.

## Minimal Python SDK usage

```python
from gossipmemo_client import GossipMemo

memory = GossipMemo("http://localhost:8765", space_id="personal")

result = memory.turn(
    "Alice told me Bob may be preparing to leave.",
    source={"provider": "hermes", "conversation_key": "chat-1", "item_id": "turn-1"},
)
print(result["message_ids"])

answer = memory.query("What do I know about Bob?", people=["Bob"])
print(answer["answer"])
```

`turn()` and `ingest()` both write to the same `POST /v1/spaces/{space_id}/turns`
endpoint; `ingest()` is a deprecated thin wrapper kept for its "single-field vs
messages list" normalization and for batches whose messages are not
necessarily user-authored (e.g. importing a whole conversation). Prefer
`turn()` for new code.

Manual memories can be corrected without erasing history through
`memory.supersede(...)` or withdrawn through `memory.retract(...)`.

Both `GossipMemo` and `AsyncGossipMemo` clients are included. If
`GOSSIPMEMO_API_KEY` is set on the server, pass the same value as `api_key` to
the client.

## Hermes integration

The plugin under [`integrations/hermes/gossipmemo`](integrations/hermes/gossipmemo)
implements Hermes' `MemoryProvider` interface. Install this project into the
Hermes Python environment, copy or link that plugin directory into Hermes'
memory plugin directory, and configure its server URL and Space as documented
in the plugin README.

Hermes sessions are retained only as Message source coordinates. They do not
partition GossipMemo's long-term memory.

For turn continuity the provider uses the SDK's `turn()` façade: it persists
the user message and returns recall plus a context bundle without invoking the
LLM query synthesizer. The latest bundle/version is cached per session; slow or
failed preparation falls back to the prior cache (or an empty block), so it
never blocks a chat turn. After completion Hermes asynchronously ingests the
assistant reply, reusing the user's idempotency key to avoid duplicate user
messages.

## How it works

### The write path is asynchronous

Writing a message is synchronous and durable. Everything that turns messages
into semantic memory happens later, off the request path.

```text
POST /v1/spaces/{space_id}/turns ─→ Messages persisted (idempotent) ─→ returns {"status": "accepted", "message_ids": [...]}
                     │
                     │  once 6 messages accumulate, or the oldest has waited 30 min
                     ↓
                extraction (LLM) ─→ Memory  (content, kind, basis, people, about_user)
                     │
                     ├─→ marks the touched Person / Relationship / UserModel stale
                     │
                     ↓
                reasoning pipeline ─→ refreshed cards, inferred Memories,
                     ↑                 Hypotheses, coverage entries, LearningGoals
                     │  at startup for whatever is stale, then daily at local midnight
                     │
                continuity reasoner ─→ rolling continuity text
                                        every ~20 new messages
```

Two consequences worth planning around: a Memory is not queryable the instant
you ingest it, and a Person card can lag a Memory by up to a day. The batch
size and timeout are tunable (`GOSSIPMEMO_EXTRACTION_BATCH_SIZE`,
`GOSSIPMEMO_EXTRACTION_BATCH_TIMEOUT_SECONDS`); the induction schedule is not.

Raw Messages are durable evidence and are never rewritten. Memories are the
durable semantic record, and support active, retracted, and superseded states,
so a correction never erases history. Person cards, Relationship cards, the
UserModel and continuity are **regenerable projections** over active Memories:
deleting them and rebuilding them loses nothing.

Coverage entries are the one exception. They are freely rewritten as
understanding accumulates, so rebuilding them depends on the order the audits
ran — a rebuild yields a different but equally valid set. Coverage is therefore
**replayable accumulated state**, not a regenerable projection. Nothing
unrecoverable rides on it: the evidence layer underneath is intact.

### The reasoners

Each reasoner is one LLM call type with one owned output. They are independent;
none of them reads another's card.

| Reasoner | Runs when | Maintains |
| --- | --- | --- |
| **Extraction** | 6 new messages, or 30 min after the oldest | Memories and People from raw text: content, `kind`, `basis`, linked People, `about_user`, validity window, explicit supersedes |
| **Person** | that Person is stale | The Person `profile_card`, plus inferred Memories and Hypotheses about them |
| **Relationship** | that Relationship is stale | `facets`, `closeness`, `tone`, `status`, `summary`, plus inferred Memories and Hypotheses |
| **UserModel** | the space's UserModel is stale | The space-level user `profile_card`, plus Hypotheses about the user |
| **Continuity** | ~20 new messages | A short rolling continuity text, the related Person IDs, and the last covered message |
| **Coverage audit** | new Memories since the last root's last audit | Coverage entries: one short summary per path under each of the twenty roots, of how well that area is understood |
| **Learning goals** | after each coverage audit | LearningGoals: up to three directions per root, reconciled into one plan; each is an askable prompt, why it is worth understanding, and its status |

The first five run as one ordered pipeline per space — Person, Relationship,
UserModel, Coverage, LearningGoals — so a card is refreshed before anything
audits what it still lacks. Continuity runs on its own message-count trigger.

Person, Relationship, and UserModel reasoning is two-staged: a projection call
rewrites the card, then an epistemic review call decides what to infer and what
to merely suspect. That split is what keeps *inference* out of the card:

- An **inferred Memory** is a conclusion the system is willing to state, stored
  as a real Memory with `basis = inferred` and links back to the source
  Memories it was derived from. It can be retracted later by the same reasoner.
- A **Hypothesis** is a suspicion that has not earned that status. It carries a
  confidence level and supporting/contradicting evidence, and stays out of the
  cards until it is promoted, rejected, superseded, or retired.

Freshness is watermark-based: a card is stale when a related Memory has a newer
`updated_at`. That covers additions, retractions, and supersedes with one
mechanism, so no reasoner needs an invalidation hook.

Extraction drains sequentially per space: one batch at a time, oldest and
least-attempted first, until nothing is pending. A batch that keeps failing is
retired after five recorded failures so one broken batch cannot spin the drain
forever — but only failures the server actually observed count. A process
killed mid-call leaves its batch untouched and it is retried on the next start.

Extraction reads only its own batch. It will infer that "don't change the specs
right before next month's release again" means the speaker dislikes late scope
changes, but it will not conclude that the speaker resists change in general —
that requires history, and belongs to reasoning.

### One provider request at a time

Every outbound provider request passes a single-permit gate, so exactly one is
in flight process-wide. The unit is one HTTP request, not one reasoner call: a
reasoner that has to split oversized context into several requests releases the
permit between them rather than holding it for the whole job.

Waiting callers are served by strict priority, FIFO within a tier:

1. **Foreground** — `query` synthesis, the one synchronous LLM path.
2. **Freshness** — extraction and continuity, which keep recent messages usable.
3. **Background** — everything else: the card, coverage, and goal reasoners.

There is deliberately no aging and no quota. During a large backfill,
extraction is *meant* to starve induction: a Person card induced from
half-extracted history would only be invalidated once extraction caught up.

When the provider signals a rate limit or an outage, the backoff is taken while
holding the permit. That is intentional global backpressure — one caller waiting
out a `Retry-After` stops every other caller from making it worse. Retries for
malformed model output work differently: the provider is healthy, so those wait
outside the gate and let other work through.

### What the agent actually receives

Reads never call an LLM. Alias matching is deterministic, recall is SQLite FTS5
over structured Person filters, and the cards are already built. Latency is
predictable and does not depend on a provider being up.

`GET /v1/spaces/{space_id}/context` returns a versioned bundle:

```json
{
  "version": "...",
  "user_model": { "profile_card": {}, "stale": false },
  "continuity": { "text": "...", "related_person_ids": ["..."] },
  "people":   [ { "id": "...", "display_name": "...", "profile_card": {} } ],
  "guidance": { "items": [] }
}
```

`people` holds the cards for the People the continuity text refers to — not
every Person in the space.

`POST /v1/spaces/{space_id}/turns` is the turn-oriented facade. It persists one
user message and returns, in the same response:

- `known_people` — People activated by deterministic alias matching on the
  message text
- `memory_recall` — a few relevant active `about_user` Memories, via FTS
- `guidance` — see below
- `context_update` — the current bundle, **only** when the caller's
  `context_version` is stale, so a warm caller pays nothing
- `context_status` — `available` or `unavailable`; enrichment is best-effort
  and a failure here never blocks the turn

`guidance` is how the system asks for what it is missing. It carries **at most
one open Hypothesis and a random three to five open or partial LearningGoals**.
The two are selected differently on purpose. A Hypothesis is a claim about a
specific owner, so the activated People already narrow it to the current
subject and the single best match is worth confirming: recency first, with
character-bigram overlap against the message as a tie-break (bigrams so that
CJK and unspaced text still rank). A LearningGoal is a long-term direction
rather than a claim, and only the agent knows what the conversation is
currently about — so the server does not rank goals at all. It samples, and the
agent decides which, if any, fit. Coverage entries themselves never leave the
server. The agent is free to act on guidance or ignore it; the Hermes
integration renders the goals with an explicit instruction to ignore them by
default, so several directions at once do not turn the chat into an interview.

Because goals are sampled fresh on every read, `guidance` is deliberately
excluded from `version`: the version tracks the durable context state, so a
warm caller is not invalidated every turn.

An empty bundle is a valid cold-start result. The current session's raw messages
carry continuity while long-term memory accumulates.

`POST /v1/spaces/{space_id}/query` is the one synchronous LLM path: it
retrieves People, Relationships, and Memories and synthesizes a written answer.
Agents in a chat loop should prefer `turns`; `query` is for asking the store a
question directly.

### Storage

SQLite is the first canonical-store adapter, combining FTS5 with structured
Person/Relationship filters; embeddings are intentionally deferred. Retrieval
indexes are regenerable projections, not part of the Memory entity. A single
process is a product invariant — SQLite plus the in-process, single-permit
provider gate — so there is deliberately no worker-count option.

The database runs in WAL mode, so readers never block behind the writer. The
cost is that the database is no longer a single file: `gossipmemo.db` is
accompanied by `gossipmemo.db-wal` and `gossipmemo.db-shm`, and the newest
writes live in the `-wal` file until a checkpoint folds them back. **Copying
`gossipmemo.db` alone from a running server silently loses recent data.** To
move or back up a database, either stop the server first and copy the file,
or copy all three files together. The server refuses to start if the
filesystem will not accept WAL — a network mount, usually — rather than
falling back silently.

### Schema migrations

`schema.sql` is versioned. The deployed database is upgraded in place on
restart; it is never deleted or rebuilt. `SqliteWorldStore.initialize()`
runs `migrate_database()` (`src/gossipmemo/migrations.py`) before applying
`schema.sql`, and startup fails loudly rather than serving a program against
a database it does not trust. Concretely, on every startup:

1. A `schema_migrations` table (`version`, `applied_at`, `description`,
   `checksum`) is read to find the database's current version. Its rows are
   an ordered, immutable history: nothing is ever updated or deleted from
   it, and each row's checksum is checked against the migration registered
   in code for that version. A missing, non-contiguous, or checksum-mismatched
   history is refused rather than trusted.
2. A brand-new, empty database file is stamped directly at the program's
   current schema version — there is no history to preserve, so nothing is
   replayed.
3. A database already at the current version is a no-op: restarting the
   container after a release with no schema change does no work.
4. A database behind the current version is upgraded: **a full SQLite
   backup is taken first**, written next to the live database as
   `.<dbfile>.pre-migration-v<N>.<timestamp>.bak` (a dotfile, in the same
   mounted data directory, so `docker cp`/volume access reaches it). If the
   backup cannot be created, migration aborts before touching the live
   database. Each pending version is then applied in one write transaction,
   which rolls back and re-raises on any failure, leaving the database at
   its last successfully-applied version.
5. A database stamped at a version newer than the running program refuses
   to start — this program build cannot safely serve it. Never downgrade a
   database by rolling the image back onto a newer schema.

**Restore path**: stop the container, replace the live `gossipmemo.db` /
`gossipmemo.db-wal` / `gossipmemo.db-shm` (or your configured
`GOSSIPMEMO_DATABASE_PATH`) with the `.bak` snapshot (a plain SQLite file —
restore it as the main database file, no `-wal`/`-shm` needed), and restart.

#### Upgrading this deployment from v1 to v2 (one-time manual step)

This repository's first deployed release predates migration history and is
treated as schema version 1. Commit `0bc92208314bd685a63bd0b8415eda65c511cea0`
on `main` is the last version-1 commit; the first commit that changes
`schema.sql` (`b3c0c33`, "Replace the coverage map with per-root coverage
entries") is what upgrades a deployed instance to version 2. Manual operator
care is only required for *this* first upgrade — every later migration is
meant to be invisible, per the rule above.

Before pulling a v2+ image onto a running v1 deployment:

1. Stop the container: `docker compose down` (or `docker stop <container>`).
2. Copy the entire mounted data directory by hand as an out-of-band safety
   net, in addition to the automatic in-app backup:
   `cp -a "$GOSSIPMEMO_DATA_DIR" "$GOSSIPMEMO_DATA_DIR.pre-v2-backup"`.
3. Pull/build the new image and start the container normally
   (`docker compose up --build`).
4. Verify the migration succeeded:
   - `docker compose logs` should show `world_start_complete` with no
     migration errors.
   - `GET /healthz` returns 200.
   - `sqlite3 "$GOSSIPMEMO_DATA_DIR/gossipmemo.db" "SELECT version, description FROM schema_migrations ORDER BY version;"`
     should list version 1 (legacy baseline) and version 2 (the coverage
     migration).
   - `sqlite3 "$GOSSIPMEMO_DATA_DIR/gossipmemo.db" "SELECT name FROM sqlite_master WHERE name = 'coverage_maps';"`
     should return nothing; `coverage_roots` and `coverage_entries` should
     exist and be non-empty for spaces with prior data.
5. Once confirmed, the manual directory copy from step 2 and the automatic
   `.gossipmemo.db.pre-migration-v1.*.bak` file in the data directory can be
   archived elsewhere or deleted.

### Code layout

The provider seam is deliberately narrow. `transport.py` is a leaf module —
the chat-completion models, the priority gate, the retry policy, and the
`LlmTransport` protocol, which is everything a reasoner is allowed to know
about the provider. `llm.py` is the one implementation of it: HTTP, retries,
and nothing else. Prompt assembly and chunking strategy belong to the
reasoners, one module each under `reasoners/`, with the budget-driven splitting
tools they share in `chunking.py`.

The point of the narrowness is that each reasoner's real path — two-stage owner
reasoning, lossy digesting, paginated evidence — is exercised by tests against
a transport double, rather than stubbed out behind a per-reasoner method.

See [data_schema.md](data_schema.md) for the table-level model and
[glossary.md](glossary.md) for the canonical vocabulary.

## License

GossipMemo is licensed under the GNU General Public License v3.0 only.
See [LICENSE](LICENSE).
