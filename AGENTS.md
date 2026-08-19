# GossipMemo Agent Guide

## Working style

- Keep the product MVP-oriented. Prefer the smallest design that solves a demonstrated problem; add abstractions and components only when real usage requires them.
- Discuss data schema and interface semantics before implementing consequential changes.
- Every change to `schema.sql` ships with a migration in `src/gossipmemo/migrations.py` (bump `CURRENT_VERSION`, add a `Migration` entry, extend `tests/test_migrations.py`). The deployed database is never deleted or rebuilt; `SqliteWorldStore.initialize()` runs `migrate_database()` first and upgrades it in place, backing it up beforehand. See README.md's "Schema migrations" section for the operator-facing contract.
- Split development into independently useful slices. Run sub-agents sequentially with concise context and medium reasoning effort, and commit each feature separately.
- Preserve simple domain language and use `ExtractedXXX` consistently for LLM extraction models.
- Verify changes with `UV_CACHE_DIR=/tmp/gossipmemo-uv-cache uv run pytest -q` and `git diff --check`.

## Product scope

GossipMemo is a local-first social-memory server for one user interacting with one agent. Hermes is the first supported agent integration. A `Space` is the long-term memory scope across chat sessions.

The current user is the fixed `user` message author and has one space-level `UserModel`; the user is not a `Person`, and there is no ego or author-to-Person binding. `Person` represents an external person mentioned in conversation. People are first-class records with aliases backed by a reverse index; ambiguous aliases are not merged automatically.

Raw `Message` rows are durable evidence. Ingestion supports batches; LLM extraction defaults to six messages per request and flushes a partial batch after 30 minutes. Messages and extracted Memories retain extraction-batch provenance.

`Memory` is the durable semantic record. It supports active, retracted, and superseded states. `about_user` selects Memories used to rebuild the UserModel. Memories link directly to People without structural person roles; roles remain expressed in natural-language content.

Person, Relationship, and UserModel cards are rebuildable projections over active Memories. Freshness uses related Memory `updated_at` watermarks, covering additions, retractions, and supersedes. Induction runs at startup for stale projections and then daily at local midnight.

Rolling continuity is a per-space projection generated asynchronously after about 20 new messages. It stores a concise continuity text, related Person IDs, and the last covered message. It summarizes ongoing threads and recent decisions rather than duplicating Person cards.

## Context flow

- `GET /v1/spaces/{space_id}/context` returns a versioned bundle containing the compact UserModel, rolling continuity, and continuity-related Person cards.
- `POST /v1/spaces/{space_id}/turns` is the single write endpoint: it persists a batch of 1-100 messages of either author and schedules background intake. When (and only when) the batch's last message is from the user, it also returns a newer context bundle if the caller's version is stale, activates known People through deterministic alias matching, and recalls a small number of relevant active `about_user` Memories through SQLite FTS, using that last message's content. A batch ending in an assistant message still persists but skips this enrichment.
- Alias matching, FTS recall, and context reads do not call an LLM. Extraction, profile induction, and continuity generation remain asynchronous.
- The Python sync and async SDKs expose `context()` and `turn()`.
- The Hermes provider caches the context bundle/version, uses the turn facade during prefetch, renders UserModel, continuity, activated Person cards, and recalled user Memories, then asynchronously ingests the assistant reply. User-message idempotency keys prevent duplicate writes across prefetch and completed-turn synchronization.

An empty context bundle is a valid cold-start result. The current session's raw messages provide immediate continuity while long-term Memories and projections accumulate.
