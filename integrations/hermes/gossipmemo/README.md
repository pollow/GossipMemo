# GossipMemo for Hermes

This plugin connects Hermes to a running GossipMemo HTTP server. GossipMemo
keeps the long-lived memory boundary at `space_id`; a Hermes `session_id` is
only retained as the source conversation key on ingested messages.

# Loading mode

This registers as a standalone Hermes plugin: copy or symlink this directory
under Hermes' `plugins/` directory and set `plugins.enabled: [gossipmemo]`
in the profile's `config.yaml`. `register(ctx)` wires the eight
`gossipmemo_*` tools through `ctx.register_tool` (so they land in Hermes'
tool registry and are `tool_search`-deferrable, unlike an always-eager
memory-provider registration) and wires `on_session_start`/`pre_llm_call`/
`post_llm_call`/`on_session_finalize` hooks to the underlying engine
(`GossipMemoMemoryProvider`). One gap today: there is no hook equivalent of
`system_prompt_block()`, so the stable (user-model/hypothesis) half only
rides along on a session's first `pre_llm_call` instead of living in the
system prompt; a later slice closes that gap via middleware.

There used to be a second loading mode, selected via `memory.provider:
gossipmemo`, that registered this engine as a Hermes `MemoryProvider`
instead. It has been removed: it had exactly one consumer on this host, and
duplicating the registration path for zero remaining users is exactly the
abstraction this repo's `AGENTS.md` rules out. Rolling back to it, if ever
needed, is `git checkout <sha> -- integrations/hermes/gossipmemo` from
before its removal, plus a gateway restart -- not a reason to keep it live.
One real casualty of the removal: `on_memory_write`, a `MemoryProvider` ABC
method Hermes calls directly on a registered provider object (not reachable
through `ctx.register_hook`), mirrored writes from Hermes' *built-in* memory
tool into GossipMemo. It has no plugin-mode equivalent and is gone along
with the provider path. In practice this cost nothing on the migrated
profile: its only trigger, `notify_memory_tool_write`, fires only when
Hermes' built-in memory tool writes, and that tool is disabled
(`memory.memory_enabled: false`) there -- but a deployment that still
relies on the built-in memory tool would need another way to mirror it.

**Never set `memory.provider: gossipmemo` and `plugins.enabled:
[gossipmemo]` at the same time on a Hermes build that still has the old
provider path available.** Hermes would import this module under two
different module names for the two paths, producing two independent engine
instances that do not share `_prefetch_cache`/`_context_cache`, so every
turn gets written to GossipMemo twice.

## Install and configure

Install the GossipMemo SDK in the Hermes environment, then copy or symlink
this directory to `plugins/gossipmemo/` (or install it using the normal
Hermes plugin manager), and set `plugins.enabled: [gossipmemo]`. Configure
the server with environment variables:

```shell
export GOSSIPMEMO_BASE_URL=http://127.0.0.1:8765
export GOSSIPMEMO_SPACE_ID=personal
# Optional when the server has GOSSIPMEMO_API_KEY configured:
export GOSSIPMEMO_API_KEY=change-me
```

`GOSSIPMEMO_URL` is accepted as a compatibility alias for
`GOSSIPMEMO_BASE_URL`. `base_url` and `space_id` can also be set by hand in
`$HERMES_HOME/gossipmemo.json`; API keys remain environment-only.

The plugin does not probe the network at load time. Start the server before
using its tools; request failures are returned as clear tool errors and do
not crash the agent.

## Tools

The plugin exposes eight OpenAI-style tools. All of them are deferrable, so
their schemas reach the model through `tool_search` rather than sitting in
every request's tool array.

- `gossipmemo_recall` is the default lookup: LLM-free keyword (FTS) search
  over stored memories. Reach for this first.
- `gossipmemo_dossier` reads a person or relationship projection without an
  open-ended synthesis query.
- `gossipmemo_people` lists or searches known Person records.
- `gossipmemo_guidance` lists open hypotheses and open/partial learning goals for an
  explicit ask, shuffled on every call so repeated asks walk the pool rather
  than circling its most-recently-touched head. The passive context bundle
  carries only a small, seed-stable sample by contrast.
- `gossipmemo_store` creates an explicit manual memory, including person role
  links such as `subject`, `asserter`, or `reporter`.
- `gossipmemo_retract` retracts a memory while preserving its provenance.
- `gossipmemo_merge_people` merges two confirmed Person records after the user
  has confirmed that they represent the same person.
- `gossipmemo_query` asks a question and returns a synthesized answer over
  people, relationships, memories, and optional evidence. **This calls an LLM
  to synthesize the answer**, so it is the expensive exception, not the entry
  point: reserve it for genuine questions a plain `gossipmemo_recall` cannot
  answer. The always-injected instruction block and both tool descriptions
  point at `gossipmemo_recall` first for this reason.

Completed user/assistant turns are queued to a daemon writer, so
`sync_turn()` does not wait for HTTP or server-side extraction. Writes from
non-primary Hermes contexts (for example, cron prompts or subagents) are
skipped to avoid polluting the primary person's social world.

## Known gaps

- **Non-primary session detection is a heuristic, not a real signal.**
  Provider mode received an explicit `agent_context` kwarg Hermes used to
  flag cron/subagent sessions; plugin hooks carry no such kwarg. `_is_primary_session`
  reconstructs the guard from Hermes' `cron_<job_id>_<timestamp>` session-id
  naming convention, which catches cron sessions but has no signal at all
  for subagent sessions -- a subagent's turns can still be ingested as the
  primary user's memories.
- **Session-id rotation is inferred, not announced.** Hermes silently swaps
  the `session_id` a hook call carries when context compression rotates it
  mid-conversation (and on `/resume` and `/branch`), with no hook telling
  this plugin it happened. Because `_prefetch_cache`,
  `_current_turn`, and `_stable_delivered` are all keyed on `session_id`,
  relying only on Hermes' `is_first_turn` flag to decide when to (re-)deliver
  the stable (user-model/hypothesis) block would mean an unannounced
  rotation's new key never gets it for the rest of that conversation --
  Hermes never sets `is_first_turn` again once a session is underway. The
  fix is defensive rather than a recovered hook: `pre_llm_call` treats any
  session key it has not seen this session as a first turn in its own
  right (`GossipMemoMemoryProvider._claim_first_turn`), independent of
  `is_first_turn`, so compression, `/resume`, and `/branch` are all covered
  the same way without depending on any of them being announced.
