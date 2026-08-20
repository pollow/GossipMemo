from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import Settings
from .embedding import (
    DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
    EmbeddingClient,
    create_embedding_client,
    embed_query_vector,
)
from .models import (
    HealthResponse,
    ManualMemoryRequest,
    MergePersonResponse,
    MessageInput,
    QueryRequest,
    QueryResponse,
    QueueStatus,
    SupersedeRequest,
    TurnRequest,
    TurnResponse,
)
from .query import synthesize
from .reasoners import (
    Reasoner,
    ReasoningSettings,
    build_continuity_reasoner,
    build_coverage_reasoner,
    build_extraction_reasoner,
    build_learning_goals_reasoner,
    build_person_reasoner,
    build_relationship_reasoner,
    build_user_model_reasoner,
)
from .reasoning import DEFAULT_REASONING_PIPELINE, ReasoningPipeline
from .store import SqliteWorldStore, TurnView
from .store._vectors import EmbeddingUpsert
from .transport import LlmTransport

logger = logging.getLogger(__name__)

# Module-level singleton so the default is shared rather than rebuilt per call.
DEFAULT_REASONING = ReasoningSettings()

# One request carries this many (owner_kind, owner_id) texts to the
# embedding provider -- within the brief's 32-64 band. `pending_embeddings()`
# already anti-joins across every space, so a single global worker task
# processes all spaces per run; per-space scoping would be pointless extra
# bookkeeping for a store call that has no space_id parameter.
_EMBEDDING_BATCH_SIZE = 48
_EMBEDDING_RETRY_BASE_SECONDS = 5.0
_EMBEDDING_RETRY_MAX_SECONDS = 120.0


class SocialMemoryWorld:
    """Deep social-memory module used by HTTP callers and tests.

    Persistence details are internal implementation. Ingest is intentionally
    eventual: it durably records Message inputs, then returns after their
    Extract work has been scheduled as a background task. Provider-side
    serialization and priority live in `transport.ProviderGate`, not here.
    """

    def __init__(
        self,
        store: SqliteWorldStore,
        model: LlmTransport,
        extraction_batch_size: int = 6,
        extraction_batch_timeout_seconds: float = 1800.0,
        induction_interval_seconds: float | None = None,
        induction_time: str = "00:00",
        continuity_threshold: int = 20,
        reasoning: ReasoningSettings = DEFAULT_REASONING,
        settings: Settings | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.extraction_batch_size = extraction_batch_size
        self.extraction_batch_timeout_seconds = extraction_batch_timeout_seconds
        self.induction_interval_seconds = induction_interval_seconds
        self.induction_time = induction_time
        self.continuity_threshold = continuity_threshold
        self.reasoning = reasoning
        # `settings` drives async, network-probing client construction in
        # `start()` (dimension discovery talks to the provider) -- it cannot
        # happen here, in a sync constructor. Tests that don't care about
        # that probe pass a ready-made `embedding_client` directly instead,
        # which `start()` then leaves alone. Neither set means the embedding
        # subsystem is off and every path below falls back to plain FTS.
        self._settings = settings
        self._embedding_client: EmbeddingClient | None = embedding_client
        reasoners: dict[str, Reasoner] = {
            "person": build_person_reasoner(self.store, self.model, reasoning),
            "relationship": build_relationship_reasoner(self.store, self.model, reasoning),
            "user_model": build_user_model_reasoner(self.store, self.model, reasoning),
            "coverage": build_coverage_reasoner(self.store, self.model, reasoning),
            "learning_goals": build_learning_goals_reasoner(self.store, self.model, reasoning),
        }
        self._continuity_reasoner: Reasoner = build_continuity_reasoner(
            self.store, self.model, reasoning)
        self._extraction_reasoner: Reasoner = build_extraction_reasoner(
            self.store, self.model, reasoning,
            embedding_client_getter=lambda: self._embedding_client,
            embedding_query_timeout_seconds=(
                settings.embedding_query_timeout_seconds if settings is not None
                else DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS
            ),
        )
        self.reasoning_pipeline = ReasoningPipeline(
            [reasoners[name] for name in DEFAULT_REASONING_PIPELINE],
            should_continue=lambda: not self._stopping,
        )
        self._tasks: set[asyncio.Task[Any]] = set()
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._scheduled: set[tuple[str, str, str]] = set()
        self._dirty_intake_spaces: set[str] = set()
        self._stopping = False
        self._induction_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        logger.info("world_start_begin")
        started = asyncio.get_running_loop().time()
        self._stopping = False
        self.store.initialize()
        await self._start_embedding_subsystem()
        unbatched_spaces: set[str] = set()
        for pending in self.store.pending_extractions():
            if pending.batch_id:
                self._schedule_extraction(pending.space_id)
            else:
                unbatched_spaces.add(pending.space_id)
        for space_id in unbatched_spaces:
            self._drain_extraction_batches(space_id)
        self._schedule_all_stale()
        for space_id in self.store.pending_continuities(self.continuity_threshold):
            self._schedule_continuity_reason(space_id)
        self._induction_task = asyncio.create_task(
            self._induction_loop(), name="gossipmemo-daily-induction"
        )
        logger.info(
            "world_start_complete",
            extra={
                "pending_extractions": len(self.store.pending_extractions()),
                "duration_ms": round(
                    (asyncio.get_running_loop().time() - started) * 1000, 2
                ),
            },
        )

    async def _start_embedding_subsystem(self) -> None:
        """Resolve the embedding client (if not already injected) and kick
        off a full backfill. Must run after `store.initialize()` so any
        pending migration has already applied -- `ensure_vector_index()`
        reads/rebuilds the sidecar against the current main-table schema.

        Every failure mode here -- unconfigured, provider unreachable,
        dimension probe/conflict -- degrades to "embedding subsystem off"
        rather than failing startup. This is the one hard requirement from
        the design brief: an operator who never sets
        `GOSSIPMEMO_EMBEDDING_MODEL`, or whose embedding server is briefly
        down, still gets a working GossipMemo on plain FTS.
        """

        if self._embedding_client is None and self._settings is not None:
            if self._settings.embedding_enabled:
                try:
                    self._embedding_client = await create_embedding_client(self._settings)
                except Exception:
                    logger.exception("embedding_client_init_failed")
                    self._embedding_client = None
        if self._embedding_client is not None:
            self.store.ensure_vector_index(
                self._embedding_client.model, self._embedding_client.dim
            )
            self._schedule_embedding()

    async def stop(self) -> None:
        started = asyncio.get_running_loop().time()
        self._stopping = True
        if self._induction_task:
            self._induction_task.cancel()
            await asyncio.gather(self._induction_task, return_exceptions=True)
            self._induction_task = None
        flush_tasks = tuple(self._flush_tasks.values())
        for task in flush_tasks:
            task.cancel()
        if flush_tasks:
            await asyncio.gather(*flush_tasks, return_exceptions=True)
        self._flush_tasks.clear()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        await self.model.aclose()
        await self._close_embedding_client()
        logger.info("world_stop_complete", extra={"duration_ms": round(
            (asyncio.get_running_loop().time() - started) * 1000, 2)})

    async def _close_embedding_client(self) -> None:
        """Release the embedding client's resources on shutdown.

        `EmbeddingClient` deliberately does not declare `aclose` on its
        Protocol -- a minimal fake (`FakeEmbeddingClient`, used throughout
        the test suite) has no connection pool to release and should not
        have to grow a no-op method just to satisfy this call, so the call
        is protective (`getattr`) rather than assumed.

        Whether world "owns" the client is not decided by *how* World got
        it (constructor injection vs. built from `settings` in `start()`):
        `self.model` -- always constructed by the caller, sometimes a test
        double -- is closed unconditionally the same way just above, and an
        embedding client follows that same precedent for consistency. The
        thing `OpenAICompatibleEmbeddingClient._owns_client` decides is a
        level below this: whether *its* `aclose()` actually tears down the
        underlying `httpx.AsyncClient`, or leaves alone one the caller
        supplied to its own constructor. Calling `aclose()` here is always
        safe either way -- it is that method's job to know which.
        """

        client = self._embedding_client
        if client is None:
            return
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()

    def _spawn(
        self,
        key: tuple[str, str, str],
        operation: Coroutine[Any, Any, None],
        *,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        """Run `operation` as a deduplicated background task.

        `on_success` fires only after `operation` completes without raising
        -- never on the dedup-skip path and never after a failed attempt
        (that reasoner will simply be retried on its own next trigger).
        Deliberately a callback rather than a second coroutine wrapped
        around `operation`: wrapping would mean the dedup-skip branch below
        closes the *wrapper*, not `operation` itself, and since an
        unstarted coroutine's `.close()` does not run any of its body, the
        real `operation` coroutine would never get closed and would be
        garbage-collected instead -- a "coroutine was never awaited" leak.
        A plain callback keeps `operation` exactly what gets passed to
        `.close()` here, matching every other `_spawn` call site.
        """

        if self._stopping or key in self._scheduled:
            operation.close()
            return
        self._scheduled.add(key)

        async def run() -> None:
            try:
                await operation
            except Exception:
                logger.exception("background memory operation failed: %s", key)
            else:
                if on_success is not None:
                    on_success()
            finally:
                self._scheduled.discard(key)

        task = asyncio.create_task(run(), name="gossipmemo-" + "-".join(key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _embedding_query_timeout_seconds(self) -> float:
        if self._settings is not None:
            return self._settings.embedding_query_timeout_seconds
        return DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS

    async def _embed_query(self, text: str, instruction: str) -> list[float] | None:
        """Best-effort query-side embedding for a hybrid recall call site.

        Delegates the actual degrade discipline (no client, request/protocol
        error, timeout) to `embedding.embed_query_vector` -- this method only
        supplies this world's client and its short, independent timeout
        (never `llm_timeout_seconds`; see `Settings.embedding_query_timeout_seconds`).
        `None` here always means "the caller falls back to plain FTS."
        """
        return await embed_query_vector(
            self._embedding_client, text,
            instruction=instruction, timeout=self._embedding_query_timeout_seconds(),
        )

    def merge_person(
        self, space_id: str, source_person_id: str, target_person_id: str
    ) -> MergePersonResponse:
        return MergePersonResponse.model_validate(
            self.store.merge_person(space_id, source_person_id, target_person_id)
        )

    async def import_messages(
        self, space_id: str, messages: list[MessageInput]
    ) -> dict[str, int]:
        """Durably import and run every stage for this space to completion.

        Unlike the server path this is synchronous by design: a CLI import
        drives each reasoner directly, in order, and lets failures raise.
        """
        self.store.initialize()
        message_ids = self.store.record_messages(space_id, messages)
        # Imports must not leave a partial six-message batch waiting for the timer.
        while True:
            pending = self.store.unbatched_messages(space_id)
            if not pending:
                break
            batch_id = self.store.create_extraction_batch(
                space_id,
                [message_id for message_id, _ in pending[: self.extraction_batch_size]],
            )
            if not batch_id:
                break
        await self._catch_up(self._extraction_reasoner, space_id)
        states = self.store.extraction_states(space_id, message_ids)
        if "failed" in states:
            raise RuntimeError("one or more imported messages failed extraction")
        if any(state != "completed" for state in states):
            raise RuntimeError("imported message extraction did not complete")
        # Only this space: an import must not do work for unrelated spaces,
        # and only when stale, so a repeated import stays a no-op.
        if space_id in self._stale_spaces():
            await self.reasoning_pipeline.run_until_caught_up(space_id)
        if self.store.pending_continuities(self.continuity_threshold, space_id):
            await self._catch_up(self._continuity_reasoner, space_id)
        # Best-effort: an import must not fail because the embedding
        # provider is unavailable, and world.stop() below still waits for
        # this task, so a healthy provider leaves the import fully embedded
        # by the time the CLI prints its summary.
        self._schedule_embedding()
        return {
            "messages": len(message_ids),
            "extracted": len(message_ids),
        }

    async def turn(self, space_id: str, request: TurnRequest) -> TurnResponse:
        """Persist this batch first; all enrichment is best-effort and local.

        Every batch records messages and schedules intake identically. Only
        read-enrichment (alias matching, guidance, FTS recall,
        context-version comparison) is conditional, and the condition is
        derived from the data itself -- the batch's last message is from the
        user, or it is not -- never from a caller-supplied flag.
        """
        started = asyncio.get_running_loop().time()
        message_ids = self.store.record_messages(space_id, request.messages)
        self._schedule_intake(space_id)
        last_message = request.messages[-1]
        view = TurnView()
        if last_message.author == "user":
            query_vector = await self._embed_query(
                last_message.content, self.reasoning.prompts.embedding_turn_recall_instruction,
            )
            view = self.store.turn_view(
                space_id, last_message.content, request.memory_limit, request.context_version,
                query_vector,
            )
        logger.info("turn_completed", extra={
            "space_id": space_id,
            "message_count": len(message_ids),
            "known_people": len(view.known_people),
            "recalled_memories": len(view.memory_recall),
            "context_status": view.context_status,
            "duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
        })
        return TurnResponse(
            message_ids=message_ids,
            known_people=view.known_people,
            memory_recall=view.memory_recall,
            guidance=view.guidance,
            context_update=view.context_update,
            context_status=view.context_status,
        )

    def _schedule_intake(self, space_id: str) -> None:
        """Move batch-drain + continuity scheduling off the request path."""
        self._dirty_intake_spaces.add(space_id)
        self._spawn(("intake", space_id, space_id), self._intake_loop(space_id))

    async def _intake_loop(self, space_id: str) -> None:
        # Loop again if new messages marked this space dirty while we were
        # draining, so nothing arriving between the check and task end is
        # missed. The 30-minute partial flush remains the final backstop.
        while True:
            self._dirty_intake_spaces.discard(space_id)
            self._drain_extraction_batches(space_id)
            if self.store.pending_continuities(self.continuity_threshold, space_id):
                self._schedule_continuity_reason(space_id)
            if space_id not in self._dirty_intake_spaces:
                break

    def _catch_up(self, reasoner: Reasoner, space_id: str) -> Coroutine[Any, Any, None]:
        # The driver contributes only the stop predicate; the reasoner owns
        # its own loop and everything inside one unit (load, call, commit).
        return reasoner.run_until_caught_up(space_id, lambda: not self._stopping)

    def _schedule_continuity_reason(self, space_id: str) -> None:
        logger.info("continuity_scheduled", extra={"space_id": space_id})
        self._spawn(
            ("continuity", space_id, space_id),
            self._catch_up(self._continuity_reasoner, space_id),
        )

    def _drain_extraction_batches(self, space_id: str) -> None:
        pending = self.store.unbatched_messages(space_id)
        while len(pending) >= self.extraction_batch_size:
            batch_id = self.store.create_extraction_batch(
                space_id,
                [message_id for message_id, _ in pending[: self.extraction_batch_size]],
            )
            if batch_id:
                logger.info("extraction_batch_scheduled", extra={
                    "space_id": space_id,
                    "batch_id": batch_id,
                    "message_count": self.extraction_batch_size,
                })
                self._schedule_extraction(space_id)
            pending = self.store.unbatched_messages(space_id)
        if pending:
            self._schedule_partial_flush(space_id, pending[0][1])
        else:
            flush_task = self._flush_tasks.pop(space_id, None)
            if flush_task and flush_task is not asyncio.current_task():
                flush_task.cancel()

    def _schedule_partial_flush(self, space_id: str, oldest_ingested_at: str) -> None:
        if self._stopping or space_id in self._flush_tasks:
            return
        oldest = datetime.fromisoformat(oldest_ingested_at)
        elapsed = (datetime.now(timezone.utc) - oldest).total_seconds()
        delay = max(0.0, self.extraction_batch_timeout_seconds - elapsed)

        async def flush() -> None:
            try:
                await asyncio.sleep(delay)
                self._flush_tasks.pop(space_id, None)
                pending = self.store.unbatched_messages(space_id)
                if pending:
                    batch_id = self.store.create_extraction_batch(
                        space_id,
                        [
                            message_id
                            for message_id, _ in pending[: self.extraction_batch_size]
                        ],
                    )
                    if batch_id:
                        self._schedule_extraction(space_id)
                self._drain_extraction_batches(space_id)
            finally:
                self._flush_tasks.pop(space_id, None)

        task = asyncio.create_task(flush(), name=f"gossipmemo-flush-{space_id}")
        self._flush_tasks[space_id] = task

    def _schedule_extraction(self, space_id: str) -> None:
        logger.info("extraction_scheduled", extra={"space_id": space_id})
        # `on_success` triggers the embedding worker only once extraction's
        # `Memory` rows actually land -- see `_spawn` for why this is a
        # callback rather than a second coroutine wrapped around the catch-up.
        self._spawn(
            ("extraction", space_id, space_id),
            self._catch_up(self._extraction_reasoner, space_id),
            on_success=self._schedule_embedding,
        )

    def _next_induction_delay(self) -> float:
        if self.induction_interval_seconds is not None:
            return self.induction_interval_seconds
        now = datetime.now().astimezone()
        hour_text, minute_text = self.induction_time.split(":")
        target = now.replace(
            hour=int(hour_text), minute=int(minute_text), second=0, microsecond=0
        )
        if target <= now:
            target += timedelta(days=1)
        return max(0.0, (target - now).total_seconds())

    async def _induction_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._next_induction_delay())
            if not self._stopping:
                self._schedule_all_stale()

    def _schedule_reasoning_pipeline(self, space_id: str) -> None:
        # Same reasoning as `_schedule_extraction`: the reasoning pipeline
        # writes `hypothesis`/`learning_goal`/`coverage_entry` rows, the
        # remaining owner kinds `pending_embeddings()` tracks.
        self._spawn(
            ("reasoning-pipeline", space_id, space_id),
            self.reasoning_pipeline.run_until_caught_up(space_id),
            on_success=self._schedule_embedding,
        )

    async def query(self, space_id: str, request: QueryRequest) -> QueryResponse:
        query_vector = await self._embed_query(
            request.question, self.reasoning.prompts.embedding_query_instruction,
        )
        context = self.store.read(space_id, request, query_vector)
        # `synthesize` is the only synchronous, HTTP-response-blocking call;
        # it sets the foreground gate tier itself in query.py.
        answer = await synthesize(
            self.model, request.question, context, self.reasoning.prompts)
        return QueryResponse(answer=answer, **context.model_dump())

    def add_memory(self, space_id: str, request: ManualMemoryRequest) -> str:
        return self.store.add_manual_memory(space_id, request)

    def retract_memory(
        self, space_id: str, memory_id: str, reason: str | None = None
    ) -> bool:
        changed = self.store.retract_memory(space_id, memory_id, reason)
        return changed

    def supersede_memory(
        self, space_id: str, memory_id: str, request: SupersedeRequest
    ) -> str | None:
        replacement_id = self.store.supersede_memory(space_id, memory_id, request)
        return replacement_id

    def _stale_spaces(self) -> set[str]:
        people, relationships, user_models = self.store.stale_entities()
        logger.info(
            "induction_scan_completed",
            extra={
                "people": len(people),
                "relationships": len(relationships),
                "user_models": len(user_models),
            },
        )
        stale_spaces = {space_id for space_id, _ in people}
        stale_spaces.update(space_id for space_id, _ in relationships)
        stale_spaces.update(user_models)
        # Coverage has an independent source watermark, including spaces whose
        # UserModel card is already current.
        stale_spaces.update(self.store.stale_coverage_spaces())
        return stale_spaces

    def _schedule_all_stale(self) -> None:
        if self._stopping:
            return
        for space_id in self._stale_spaces():
            self._schedule_reasoning_pipeline(space_id)

    # -- embedding worker ----------------------------------------------
    #
    # A single global background task, deduplicated through `_spawn` like
    # every other worker here. It is deliberately *not* per-space: unlike
    # extraction/continuity, `store.pending_embeddings()` has no space_id
    # parameter (it anti-joins across every space at once), so per-space
    # scheduling would just be extra bookkeeping around one shared query.
    # It is also deliberately outside `ProviderGate`: that gate exists to
    # serialize and prioritize load on the one shared remote chat provider,
    # and embedding here talks to an independent (usually local) provider.
    # Structural concurrency cap: `_spawn`'s key dedup means at most one
    # embedding-worker task is ever in flight, and within it batches are
    # requested one at a time -- that is the "small concurrency limit" the
    # design brief asks for, without a separate semaphore to maintain.

    def _schedule_embedding(self) -> None:
        if self._embedding_client is None or self._stopping:
            return
        self._spawn(("embedding", "*", "*"), self._embedding_loop())

    async def _embedding_loop(self) -> None:
        """Drain `pending_embeddings()` in batches until caught up.

        A batch failure (provider down, transport error, bad response) is
        logged and backed off -- never raised past this loop. The pending
        rows are untouched by a failed attempt, so the next trigger (the
        next turn, the next reasoner completion, or nothing at all if the
        provider stays down) simply retries them; no state is lost.
        """

        delay = _EMBEDDING_RETRY_BASE_SECONDS
        while not self._stopping:
            try:
                did_work = await self._process_embedding_batch()
            except Exception:
                logger.exception("embedding_batch_failed")
                await self._sleep_or_stop(delay)
                delay = min(delay * 2, _EMBEDDING_RETRY_MAX_SECONDS)
                continue
            delay = _EMBEDDING_RETRY_BASE_SECONDS
            if not did_work:
                break

    async def _process_embedding_batch(self) -> bool:
        """Embed and store one batch. Returns whether there was work to do."""

        client = self._embedding_client
        if client is None:
            return False
        pending = self.store.pending_embeddings(model=client.model, dim=client.dim)
        if not pending:
            return False
        batch = pending[:_EMBEDDING_BATCH_SIZE]
        # Storage-side discipline: never pass `instruction=` here. Query-side
        # asymmetric prefixing belongs to the hybrid-retrieval slice.
        vectors = await client.embed([item.text for item in batch])
        by_space: dict[str, list[EmbeddingUpsert]] = {}
        for item, vector in zip(batch, vectors, strict=True):
            by_space.setdefault(item.space_id, []).append(
                EmbeddingUpsert(
                    owner_kind=item.owner_kind, owner_id=item.owner_id,
                    text=item.text, vector=vector,
                )
            )
        for space_id, items in by_space.items():
            self.store.upsert_embeddings(space_id, client.model, client.dim, items)
        logger.info("embedding_batch_completed", extra={"count": len(batch)})
        return True

    async def _sleep_or_stop(self, seconds: float) -> None:
        """Sleep in short slices so `stop()` isn't kept waiting on a long
        backoff -- `stop()` awaits this task rather than cancelling it."""

        remaining = seconds
        while remaining > 0 and not self._stopping:
            chunk = min(1.0, remaining)
            await asyncio.sleep(chunk)
            remaining -= chunk

    def health(self) -> HealthResponse:
        gate = self.model.gate
        queue_status = QueueStatus(
            pending=gate.waiting,
            running=gate.in_flight,
            current_label=gate.current_label,
        )
        embedding_enabled = self._embedding_client is not None
        embedding_pending = (
            self.store.pending_embedding_count(
                model=self._embedding_client.model, dim=self._embedding_client.dim
            )
            if self._embedding_client is not None
            else 0
        )
        return HealthResponse(
            llm_configured=self.model.configured,
            queue=queue_status,
            embedding_enabled=embedding_enabled,
            embedding_pending=embedding_pending,
        )
