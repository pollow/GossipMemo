# GossipMemo for Hermes

This plugin connects Hermes to a running GossipMemo HTTP server. GossipMemo
keeps the long-lived memory boundary at `space_id`; a Hermes `session_id` is
only retained as the source conversation key on ingested messages.

## Install and configure

Install the GossipMemo SDK in the Hermes environment, then copy this directory
to `plugins/memory/gossipmemo/` (or install it using the normal Hermes plugin
manager). Configure the server with environment variables:

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
