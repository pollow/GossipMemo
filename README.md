# GossipMemo

GossipMemo is a local-first social memory server for agents. It keeps raw
messages, durable memories, people, and relationships separate so an agent can
remember who said what about whom without treating gossip as confirmed fact.

The first version runs as one FastAPI process with a SQLite volume and a single
FIFO queue for all LLM calls. Hermes connects over HTTP through the included
Python SDK and memory-provider plugin; GossipMemo does not run inside Hermes.

## Run with Docker

```bash
cp .env.example .env
# Set GOSSIPMEMO_LLM_MODEL and GOSSIPMEMO_LLM_API_KEY in .env.
docker compose up --build
```

The server listens on `http://localhost:8765`. Its health endpoint is
`GET /healthz`; interactive OpenAPI documentation is available at `/docs`.
SQLite data is stored in the `gossipmemo-data` Docker volume.

Only one server process should open a GossipMemo SQLite file. Do not add
multiple Uvicorn workers: the local LLM queue is deliberately process-local and
sequential.

## Run from source

```bash
uv sync --extra server --extra dev
uv run gossipmemo serve
```

Without an LLM model, raw and manual memories still work and query returns the
retrieved memories without synthesis. Automatic extraction remains retryable
until a model is configured.

## Minimal Python SDK usage

```python
from gossipmemo_client import GossipMemo

memory = GossipMemo("http://localhost:8765", space_id="personal")

receipt = memory.ingest(
    content="Alice told me Bob may be preparing to leave.",
    author={"provider": "hermes", "external_id": "me", "is_ego": True},
    source={"provider": "hermes", "conversation_key": "chat-1", "item_id": "turn-1"},
    extraction_policy="balanced",  # or conservative / comprehensive
)
memory.wait_for_message(receipt["messages"][0]["id"])

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
processing decisions.
