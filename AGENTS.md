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

Person, Relationship, and UserModel cards are rebuildable projections over Memories, maintained by folding the card forward over the Memories that changed since its own watermark rather than by re-projecting the whole history. Freshness uses related Memory `updated_at` watermarks, counted without regard to status, covering additions, retractions, and supersedes. Induction runs at startup for stale projections and then daily at local midnight.

Rolling continuity is a per-space projection generated asynchronously after about 20 new messages. It stores a concise continuity text, related Person IDs, and the last covered message. It summarizes ongoing threads and recent decisions rather than duplicating Person cards.

## Context flow

- `GET /v1/spaces/{space_id}/context` returns a versioned bundle containing the compact UserModel, rolling continuity, and continuity-related Person cards. Two optional knobs steer the learning-goal sample it carries: `goals` (how many; `0` for none, omitted for the default random three to five) and `goal_seed` (a caller-chosen seed replacing the bundle version, so a caller can hold the selection still across version bumps instead of reshuffling on every durable-context change). `POST /turns` accepts the same two as request fields and must be given the same values as the context read, or the two paths return differently sampled goals. A negative `goals` is a 422.
- `POST /v1/spaces/{space_id}/turns` is the single write endpoint: it persists a batch of 1-100 messages of either author and schedules background intake. When (and only when) the batch's last message is from the user, it also returns a newer context bundle if the caller's version is stale, activates known People through deterministic alias matching, and recalls a small number of relevant active `about_user` Memories through SQLite FTS, using that last message's content. A batch ending in an assistant message still persists but skips this enrichment.
- Alias matching, FTS recall, and context reads do not call an LLM. Extraction, profile induction, and continuity generation remain asynchronous.
- The Python sync and async SDKs expose `context()` and `turn()`, both taking `goals` and `goal_seed`.
- The Hermes plugin (`integrations/hermes/gossipmemo`, loaded via `plugins.enabled: [gossipmemo]`) caches the context bundle/version, uses the turn facade during prefetch, renders UserModel, continuity, activated Person cards, at most one tentative hypothesis, and recalled user Memories, then asynchronously ingests the assistant reply. It passes `goals=0`: learning goals are never injected into the passive per-turn context, because a random draw of long-term directions is not chosen for the current moment and injecting it required a per-turn disclaimer telling the model to ignore it. The agent pulls a direction with the `gossipmemo_guidance` tool instead, which shuffles per call. The plugin also drops goals at render time, so a server predating the `goals` parameter cannot reintroduce them. User-message idempotency keys prevent duplicate writes across prefetch and completed-turn synchronization.

An empty context bundle is a valid cold-start result. The current session's raw messages provide immediate continuity while long-term Memories and projections accumulate.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
