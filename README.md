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
`GOSSIPMEMO_EXTRACTION_BATCH_TIMEOUT_SECONDS`. The server applies one global
`GOSSIPMEMO_EXTRACTION_POLICY` (`conservative`, `balanced`, or `comprehensive`)
to every extraction batch; the default is `balanced`.

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

result = memory.ingest(
    content="Alice told me Bob may be preparing to leave.",
    author="user",
    source={"provider": "hermes", "conversation_key": "chat-1", "item_id": "turn-1"},
)
print(result["message_ids"])

answer = memory.query("What do I know about Bob?", people=["Bob"])
print(answer["answer"])
```

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
POST /ingest ─→ Message persisted (idempotent) ─→ returns {"status": "accepted"}
                     │
                     │  once 6 messages accumulate, or the oldest has waited 30 min
                     ↓
                extraction (LLM) ─→ Memory  (content, kind, basis, people, about_user)
                     │
                     ├─→ marks the touched Person / Relationship / UserModel stale
                     │
                     ↓
                reasoning pipeline ─→ refreshed cards, inferred Memories,
                     ↑                 Hypotheses, CoverageMap, LearningGoals
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
so a correction never erases history. Everything else — Person cards,
Relationship cards, the UserModel, continuity, coverage — is a **regenerable
projection** over active Memories. Deleting the projections and rebuilding them
loses nothing.

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
| **Coverage audit** | new Memories since the last audit | The CoverageMap: per-criterion coverage levels, open/closed knowledge boundaries, life periods, relationship arcs, behavioral contexts |
| **Learning goals** | after each coverage audit | LearningGoals: an askable prompt, its rationale, the boundaries it would close, and its status |

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
one open Hypothesis and at most one open or partial LearningGoal**, selected
deterministically: recency first, with character-bigram overlap against the
message as a tie-break (bigrams so that CJK and unspaced text still rank). The
CoverageMap itself never leaves the server — only the one question it justifies.
The agent is free to weave that question into its reply or ignore it.

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
cost is that the database is no longer a single file: `world.db` is
accompanied by `world.db-wal` and `world.db-shm`, and the newest writes live
in the `-wal` file until a checkpoint folds them back. **Copying `world.db`
alone from a running server silently loses recent data.** To move or back up
a database, either stop the server first and copy the file, or copy all three
files together. The server refuses to start if the filesystem will not accept
WAL — a network mount, usually — rather than falling back silently.

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
