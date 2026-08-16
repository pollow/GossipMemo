from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

from .llm import TIER_FRESHNESS, LLMModel, llm_call_tier
from .logging import elapsed_ms
from .models import (
    ContextBundle,
    GuidanceBundle,
    HealthResponse,
    IngestRequest,
    IngestResponse,
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
from .reasoners import (
    PersonReasoner,
    Reasoner,
    RelationshipReasoner,
    build_continuity_reasoner,
    build_coverage_reasoner,
    build_learning_goals_reasoner,
    build_user_model_reasoner,
)
from .reasoning import DEFAULT_REASONING_PIPELINE, FunctionStage, ReasoningPipeline
from .store import SqliteWorldStore


logger = logging.getLogger(__name__)


class SocialMemoryWorld:
    """Deep social-memory module used by HTTP callers and tests.

    Persistence details are internal implementation. Ingest is intentionally
    eventual: it durably records Message inputs, then returns after their
    Extract work has been scheduled as a background task. Provider-side
    serialization and priority live in `llm.ProviderGate`, not here.
    """

    def __init__(
        self,
        store: SqliteWorldStore,
        model: LLMModel,
        extraction_batch_size: int = 6,
        extraction_batch_timeout_seconds: float = 1800.0,
        induction_interval_seconds: float | None = None,
        continuity_threshold: int = 20,
        reasoning_pipeline_order: tuple[str, ...] = DEFAULT_REASONING_PIPELINE,
    ) -> None:
        self.store = store
        self.model = model
        self.extraction_batch_size = extraction_batch_size
        self.extraction_batch_timeout_seconds = extraction_batch_timeout_seconds
        self.induction_interval_seconds = induction_interval_seconds
        self.continuity_threshold = continuity_threshold
        reasoners: dict[str, Reasoner] = {
            "person": PersonReasoner(self.store, self.model),
            "relationship": RelationshipReasoner(self.store, self.model),
            "user_model": build_user_model_reasoner(self.store, self.model),
            "coverage": build_coverage_reasoner(self.store, self.model),
            "learning_goals": build_learning_goals_reasoner(self.store, self.model),
        }
        self._continuity_reasoner: Reasoner = build_continuity_reasoner(self.store, self.model)
        self.reasoning_pipeline = ReasoningPipeline(
            [
                FunctionStage(name, self._stage_runner(reasoners[name]))
                for name in reasoning_pipeline_order
            ]
        )
        self._tasks: set[asyncio.Task[Any]] = set()
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._scheduled: set[tuple[str, str, str]] = set()
        self._background_errors: dict[tuple[str, str, str], Exception] = {}
        self._stopping = False
        self._induction_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        logger.info("world_start_begin")
        started = asyncio.get_running_loop().time()
        self._stopping = False
        self._background_errors.clear()
        self.store.initialize()
        unbatched_spaces: set[str] = set()
        for space_id, batch_id, _ in self.store.pending_extractions():
            if batch_id:
                self._schedule_extract(space_id, batch_id)
            else:
                unbatched_spaces.add(space_id)
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
        close = getattr(self.model, "aclose", None)
        if close is not None:
            await close()
        logger.info("world_stop_complete", extra={"duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2)})

    def _spawn(
        self,
        key: tuple[str, str, str],
        operation: Coroutine[Any, Any, None],
    ) -> None:
        if self._stopping or key in self._scheduled:
            operation.close()
            return
        self._scheduled.add(key)

        async def run() -> None:
            try:
                await operation
            except Exception as error:
                self._background_errors[key] = error
                logger.exception("background memory operation failed: %s", key)
            finally:
                self._scheduled.discard(key)

        task = asyncio.create_task(run(), name="gossipmemo-" + "-".join(key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def ingest(self, space_id: str, request: IngestRequest) -> IngestResponse:
        started = asyncio.get_running_loop().time()
        message_ids = self.store.record_messages(space_id, request.messages)
        self._drain_extraction_batches(space_id)
        if space_id in self.store.pending_continuities(self.continuity_threshold):
            self._schedule_continuity_reason(space_id)
        logger.info("ingest_completed", extra={"space_id": space_id, "message_count": len(message_ids), "duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2)})
        return IngestResponse(message_ids=message_ids)

    def merge_person(
        self, space_id: str, source_person_id: str, target_person_id: str
    ) -> MergePersonResponse:
        return MergePersonResponse.model_validate(
            self.store.merge_person(space_id, source_person_id, target_person_id)
        )

    async def import_messages(
        self, space_id: str, messages: list[MessageInput]
    ) -> dict[str, int]:
        """Durably import and synchronously drain extraction for CLI imports."""
        self.store.initialize()
        message_ids = self.store.record_messages(space_id, messages)
        # Imports must not leave a partial six-message batch waiting for the timer.
        while True:
            pending = self.store.unbatched_messages(space_id)
            if not pending:
                break
            batch_id = self.store.create_extraction_batch(
                space_id,
                [
                    message_id
                    for message_id, _ in pending[: self.extraction_batch_size]
                ],
            )
            if batch_id:
                self._schedule_extract(space_id, batch_id)
            else:
                break
        while True:
            extraction_running = any(
                key[0] == "extract" and key[1] == space_id
                for key in self._scheduled
            )
            if not extraction_running:
                break
            tasks = tuple(self._tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                break
        states = self.store.extraction_states(space_id, message_ids)
        if "failed" in states:
            raise RuntimeError("one or more imported messages failed extraction")
        if any(state != "completed" for state in states):
            raise RuntimeError("imported message extraction did not complete")
        self._schedule_all_stale()
        if space_id in self.store.pending_continuities(self.continuity_threshold):
            self._schedule_continuity_reason(space_id)
        # Wait for induction spawned by the imported Memories as well.
        while any(
            key[1] == space_id
            and key[0] in {"continuity", "reasoning-pipeline"}
            for key in self._scheduled
        ):
            tasks = tuple(self._tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                await asyncio.sleep(0)
        failures = [
            (key, error)
            for key, error in self._background_errors.items()
            if key[1] == space_id
        ]
        for key, _ in failures:
            self._background_errors.pop(key, None)
        if failures:
            key, error = failures[0]
            raise RuntimeError(
                f"background import operation {key[0]} failed: {error}"
            ) from error
        return {
            "messages": len(message_ids),
            "extracted": len(message_ids),
        }

    async def turn(self, space_id: str, request: TurnRequest) -> TurnResponse:
        """Persist this turn first; all enrichment is best-effort and local."""
        started = asyncio.get_running_loop().time()
        message_ids = self.store.record_messages(space_id, [request.message])
        self._drain_extraction_batches(space_id)
        if space_id in self.store.pending_continuities(self.continuity_threshold):
            self._schedule_continuity_reason(space_id)
        message_id = message_ids[0]
        known_people = []
        memory_recall = []
        context_update: ContextBundle | None = None
        guidance = GuidanceBundle()
        context_status = "available"
        try:
            known_people = self.store.match_people_in_text(space_id, request.message.content)
        except Exception:
            context_status = "unavailable"
            logger.exception("turn person matching failed for %s", space_id)
        try:
            guidance = self.store.guidance_bundle(
                space_id, [person.id for person in known_people], request.message.content
            )
        except Exception:
            context_status = "unavailable"
            logger.exception("turn guidance preparation failed for %s", space_id)
        try:
            memory_recall = self.store.recall_user_memories(
                space_id, request.message.content, request.memory_limit
            )
        except Exception:
            logger.exception("turn memory recall failed for %s", space_id)
        try:
            latest = self.store.context_bundle(space_id)
            if latest.version != request.context_version:
                context_update = latest
        except Exception:
            context_status = "unavailable"
            logger.exception("turn context preparation failed for %s", space_id)
        logger.info("turn_completed", extra={"space_id": space_id, "message_count": 1, "known_people": len(known_people), "recalled_memories": len(memory_recall), "context_status": context_status, "duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2)})
        return TurnResponse(
            message_id=message_id,
            known_people=known_people,
            memory_recall=memory_recall,
            guidance=guidance,
            context_update=context_update,
            context_status=context_status,
        )

    def _stage_runner(self, reasoner: Reasoner) -> Callable[[str], Coroutine[Any, Any, None]]:
        # The driver owns this loop and the `_stopping` check; the reasoner
        # owns everything inside one `attempt` (load, call, commit).
        async def run(space_id: str) -> None:
            while not self._stopping and await reasoner.attempt(space_id):
                pass

        return run

    def _schedule_continuity_reason(self, space_id: str) -> None:
        logger.info("continuity_scheduled", extra={"space_id": space_id})
        self._spawn(
            ("continuity", space_id, space_id),
            self._stage_runner(self._continuity_reasoner)(space_id),
        )

    def _drain_extraction_batches(self, space_id: str) -> None:
        pending = self.store.unbatched_messages(space_id)
        while len(pending) >= self.extraction_batch_size:
            batch_id = self.store.create_extraction_batch(
                space_id,
                [message_id for message_id, _ in pending[: self.extraction_batch_size]],
            )
            if batch_id:
                logger.info("extraction_batch_scheduled", extra={"space_id": space_id, "batch_id": batch_id, "message_count": self.extraction_batch_size})
                self._schedule_extract(space_id, batch_id)
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
                        self._schedule_extract(space_id, batch_id)
                self._drain_extraction_batches(space_id)
            finally:
                self._flush_tasks.pop(space_id, None)

        task = asyncio.create_task(flush(), name=f"gossipmemo-flush-{space_id}")
        self._flush_tasks[space_id] = task

    def _schedule_extract(self, space_id: str, batch_id: str) -> None:
        logger.info("extraction_scheduled", extra={"space_id": space_id, "batch_id": batch_id})
        self._spawn(
            ("extract", space_id, batch_id),
            self._extract(space_id, batch_id),
        )

    async def _extract(self, space_id: str, batch_id: str) -> None:
        messages = self.store.load_batch(space_id, batch_id)
        if not messages:
            return
        context = self.store.load_extraction_context(space_id, batch_id)
        known_people = self.store.load_known_people(space_id, messages + context)
        comparisons = self.store.load_extraction_comparisons(space_id, batch_id)
        started = asyncio.get_running_loop().time()
        self.store.mark_extraction_attempt(space_id, batch_id)
        try:
            with llm_call_tier(TIER_FRESHNESS):
                result = await self.model.extract(
                    messages, context, known_people, comparisons,
                )
            self.store.apply_extraction(
                space_id, batch_id, result,
                {memory.id for memory in comparisons},
            )
            logger.info("extraction_completed", extra={"space_id": space_id, "batch_id": batch_id, "message_count": len(messages), "duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2)})
        except Exception as error:
            self.store.fail_extraction(space_id, batch_id, str(error))
            logger.exception("extract failed for %s", batch_id)
            return
        if self._stopping:
            return

    def _next_induction_delay(self) -> float:
        if self.induction_interval_seconds is not None:
            return self.induction_interval_seconds
        now = datetime.now().astimezone()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(0.0, (tomorrow - now).total_seconds())

    async def _induction_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._next_induction_delay())
            if not self._stopping:
                self._schedule_all_stale()

    def _schedule_reasoning_pipeline(self, space_id: str) -> None:
        self._spawn(
            ("reasoning-pipeline", space_id, space_id),
            self.reasoning_pipeline.run_until_caught_up(space_id),
        )

    async def query(self, space_id: str, request: QueryRequest) -> QueryResponse:
        context = self.store.read(space_id, request)
        # `synthesize` is the only synchronous, HTTP-response-blocking call;
        # it sets the foreground gate tier itself in llm.py.
        answer = await self.model.synthesize(request.question, context)
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

    def _schedule_all_stale(self) -> None:
        if self._stopping:
            return
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
        for space_id in stale_spaces:
            self._schedule_reasoning_pipeline(space_id)

    def health(self) -> HealthResponse:
        # Fakes used in tests need not expose a gate; report an idle queue
        # when one is absent instead of requiring every LlmModel to have one.
        gate = getattr(self.model, "gate", None)
        if gate is None:
            queue_status = QueueStatus(pending=0, running=False, current_label=None)
        else:
            queue_status = QueueStatus(
                pending=gate.waiting,
                running=gate.in_flight,
                current_label=gate.current_label,
            )
        return HealthResponse(
            llm_configured=self.model.configured,
            queue=queue_status,
        )
