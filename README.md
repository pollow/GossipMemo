# GossipMemo

GossipMemo is a local-first social memory server for agents. It keeps raw
messages, durable memories, people, and relationships separate so an agent can
remember who said what about whom without treating gossip as objectively true.

The first version runs as one FastAPI process with a SQLite volume and a single
FIFO queue for all LLM calls. Hermes connects over HTTP through the included
Python SDK and memory-provider plugin; GossipMemo does not run inside Hermes.

## Run with Docker

```bash
cp .env.example .env
# Set GOSSIPMEMO_LLM_BASE_URL, GOSSIPMEMO_LLM_MODEL, and
# GOSSIPMEMO_LLM_API_KEY in .env.
docker compose up --build
```

The server listens on `http://localhost:8765`. Its health endpoint is
`GET /healthz`; interactive OpenAPI documentation is available at `/docs`.
SQLite data is stored in the `gossipmemo-data` Docker volume.

Useful container commands:

```bash
docker compose up -d --build       # start the single server process
docker compose ps                  # show health status
docker compose logs -f gossipmemo  # follow JSON application logs
curl -fsS http://127.0.0.1:8765/healthz
docker compose run --rm \
  -v "$PWD/export.jsonl:/imports/export.jsonl:ro" \
  gossipmemo gossipmemo import --space personal --chat /imports/export.jsonl
```

The import bind mount is read-only by design; make sure the host file is
readable by the container's non-root user. The named database volume is owned
by the fixed `gossipmemo` user (UID 10001). For a host directory bind mount at
`/data`, create it first and grant it to UID 10001, for example:
`mkdir -p ./gossipmemo-data && sudo chown 10001:10001 ./gossipmemo-data`, then
replace the named volume with `./gossipmemo-data:/data`.

Only one server process should open a GossipMemo SQLite file. Do not add
multiple Uvicorn workers: the local LLM queue is deliberately process-local and
sequential.

## Run from source

```bash
uv sync --extra server --extra dev
cp .env.example .env
# Fill all three required GOSSIPMEMO_LLM_* values, then:
uv run --env-file .env gossipmemo serve
```

Server startup is strict: `GOSSIPMEMO_LLM_BASE_URL`,
`GOSSIPMEMO_LLM_MODEL`, and `GOSSIPMEMO_LLM_API_KEY` must all be non-empty.
There is no default provider URL or no-LLM query fallback. Configuration is
read once at process initialization and passed to the server modules as one
immutable `Settings` value.

`GOSSIPMEMO_USER_NAME` names the fixed current user in LLM prompts (default:
`CurrentUser`). The current user is not stored as a `Person`.

Extraction batches wait for 6 messages by default. A partial batch is flushed
when its oldest message has waited 30 minutes. These values can be changed with
`GOSSIPMEMO_EXTRACTION_BATCH_SIZE` and
`GOSSIPMEMO_EXTRACTION_BATCH_TIMEOUT_SECONDS`. The server applies one global
`GOSSIPMEMO_EXTRACTION_POLICY` (`conservative`, `balanced`, or `comprehensive`)
to every extraction batch; the default is `balanced`.

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

## Architecture

The external `SocialMemoryWorld` interface has three behaviors:

```text
ingest(messages)  -> durably record and queue extraction
query(request)    -> retrieve and synthesize social context
apply(change)     -> manual memory and corrections
```

SQLite is the first canonical-store Adapter. The first version combines FTS5
with structured Person/Relationship filters; embeddings are intentionally
deferred. Retrieval indexes are regenerable projections, not part of the
Memory entity.
See [data_schema.md](data_schema.md) and [design.md](design.md) for the model and
processing decisions. See [glossary.md](glossary.md) for the canonical domain
and design vocabulary.

## License

GossipMemo is licensed under the GNU General Public License v3.0 only.
See [LICENSE](LICENSE).
