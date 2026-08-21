# GossipMemo for Hermes

This plugin connects Hermes to a running GossipMemo HTTP server. GossipMemo
keeps the long-lived memory boundary at `space_id`; a Hermes `session_id` is
only retained as the source conversation key on ingested messages.

## Loading modes

`register(ctx)` detects which kind of context Hermes hands it and adapts:

- **Memory-provider mode** (`plugins/memory/gossipmemo/`, selected via
  `memory.provider: gossipmemo`): registers a `MemoryProvider`. Its eight
  tool schemas are appended directly onto every request and are always
  eager (not `tool_search`-deferrable), because the stub context Hermes
  uses for this path has no tool registry to defer them from.
- **Plugin mode** (`plugins/gossipmemo/`, opted into like any other
  standalone plugin): registers the same tools through `ctx.register_tool`,
  so they land in Hermes' tool registry and become `tool_search`-deferrable,
  and wires `on_session_start`/`pre_llm_call`/`post_llm_call`/
  `on_session_finalize` hooks to the same underlying engine. One gap today:
  there is no hook equivalent of `system_prompt_block()`, so the stable
  (user-model/hypothesis) half only rides along on a session's first
  `pre_llm_call` instead of living in the system prompt; a later slice
  closes that gap via middleware.

Both modes share one engine (`GossipMemoMemoryProvider`) and are functionally
equivalent otherwise, including the non-primary-session write guard below.

## Install and configure

Install the GossipMemo SDK in the Hermes environment, then copy this directory
to `plugins/memory/gossipmemo/` for memory-provider mode, or to
`plugins/gossipmemo/` for plugin mode (or install it using the normal Hermes
plugin manager). Configure the server with environment variables:

```shell
export GOSSIPMEMO_BASE_URL=http://127.0.0.1:8765
export GOSSIPMEMO_SPACE_ID=personal
# Optional when the server has GOSSIPMEMO_API_KEY configured:
export GOSSIPMEMO_API_KEY=change-me
```

`GOSSIPMEMO_URL` is accepted as a compatibility alias for
`GOSSIPMEMO_BASE_URL`. The Hermes setup wizard can also save `base_url` and
`space_id` to `$HERMES_HOME/gossipmemo.json`; API keys remain environment-only.

The provider does not probe the network from `is_available()`. Start the
server before using its tools; request failures are returned as clear tool
errors and do not crash the agent.

## Tools

The provider exposes five OpenAI-style tools:

- `gossipmemo_query` asks a question and returns a synthesized answer over
  people, relationships, memories, and optional evidence. This calls an LLM
  to synthesize the answer, so it is reserved for genuine questions rather
  than routine lookups.
- `gossipmemo_store` creates an explicit manual memory, including person role
  links such as `subject`, `asserter`, or `reporter`.
- `gossipmemo_dossier` reads a person or relationship projection without an
  open-ended synthesis query.
- `gossipmemo_retract` retracts a memory while preserving its provenance.
- `gossipmemo_merge_people` merges two confirmed Person records after the user
  has confirmed that they represent the same person.

Completed user/assistant turns are queued to a daemon writer, so
`sync_turn()` does not wait for HTTP or server-side extraction. Writes from
non-primary Hermes contexts (for example, cron prompts or subagents) are
skipped to avoid polluting the primary person's social world.
