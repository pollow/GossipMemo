from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

from .llm import LLMModel
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
    SupersedeRequest,
    TurnRequest,
    TurnResponse,
)
from .queue import SequentialLLMQueue
from .reasoning import DEFAULT_REASONING_PIPELINE, FunctionStage, ReasoningPipeline
from .store import SqliteWorldStore


logger = logging.getLogger(__name__)


class SocialMemoryWorld:
    """Deep social-memory module used by HTTP callers and tests.

    The queue and persistence details are internal implementation. Ingest is
    intentionally eventual: it durably records Message inputs, then returns
    after their Extract work has entered the local FIFO queue.
    """

    def __init__(
        self,
        store: SqliteWorldStore,
        model: LLMModel,
        queue: SequentialLLMQueue | None = None,
        extraction_batch_size: int = 6,
        extraction_batch_timeout_seconds: float = 1800.0,
        induction_interval_seconds: float | None = None,
        continuity_threshold: int = 20,
        reasoning_pipeline_order: tuple[str, ...] = DEFAULT_REASONING_PIPELINE,
    ) -> None:
        self.store = store
        self.model = model
        self.queue = queue or SequentialLLMQueue()
        self.extraction_batch_size = extraction_batch_size
        self.extraction_batch_timeout_seconds = extraction_batch_timeout_seconds
        self.induction_interval_seconds = induction_interval_seconds
        self.continuity_threshold = continuity_threshold
        runners = {
            "person": self._run_person_stage,
            "relationship": self._run_relationship_stage,
            "user_model": self._run_user_model_stage,
            "coverage": self._run_coverage_stage,
            "learning_goals": self._run_learning_goals_stage,
        }
        self.reasoning_pipeline = ReasoningPipeline(
            [FunctionStage(name, runners[name]) for name in reasoning_pipeline_order]
        )
        self._tasks: set[asyncio.Task[Any]] = set()
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._scheduled: set[tuple[str, str, str]] = set()
        self._stopping = False
        self._induction_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        logger.info("world_start_begin")
        started = asyncio.get_running_loop().time()
        self._stopping = False
        self.store.initialize()
        await self.queue.start()
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
        await self.queue.stop()
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
            except Exception:
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

    def _schedule_continuity_reason(self, space_id: str) -> None:
        logger.info("continuity_scheduled", extra={"space_id": space_id})
        self._spawn(("continuity", space_id, space_id), self._reason_continuity(space_id))

    async def _reason_continuity(self, space_id: str) -> None:
        while not self._stopping:
            context = self.store.continuity_context(space_id)
            if not context:
                return
            continuity, messages = context
            if not messages:
                return
            result = await self.queue.submit(
                "reason-continuity", self.model.reason_continuity, continuity, messages
            )
            logger.info("continuity_extracted", extra={"space_id": space_id, "message_count": len(messages)})
            expected = continuity.through_message_id if continuity else None
            if self.store.apply_continuity_reasoning(space_id, expected, result):
                continue
            return

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
            result = await self.queue.submit(
                "extract", self.model.extract, messages, context, known_people,
                comparisons,
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

    async def _run_person_stage(self, space_id: str) -> None:
        while not self._stopping:
            people, _, _ = self.store.stale_entities()
            targets = [(sid, pid) for sid, pid in people if sid == space_id]
            if not targets:
                return
            for _, person_id in targets:
                await self._reason_person(space_id, person_id)

    async def _run_relationship_stage(self, space_id: str) -> None:
        while not self._stopping:
            _, relationships, _ = self.store.stale_entities()
            targets = [(sid, rid) for sid, rid in relationships if sid == space_id]
            if not targets:
                return
            for _, relationship_id in targets:
                await self._reason_relationship(space_id, relationship_id)

    async def _run_user_model_stage(self, space_id: str) -> None:
        while not self._stopping:
            _, _, user_models = self.store.stale_entities()
            if space_id not in user_models:
                return
            await self._reason_user_model(space_id)

    async def _run_coverage_stage(self, space_id: str) -> None:
        if not self._stopping and space_id in self.store.stale_coverage_spaces():
            await self._reason_coverage(space_id)

    async def _run_learning_goals_stage(self, space_id: str) -> None:
        await self._plan_learning_goals(space_id)

    def _owner_reason_args(self, method: object, modern: tuple[object, ...], legacy_count: int) -> tuple[object, ...]:
        """Keep pre-two-stage deterministic fakes usable during the seam transition."""
        parameters = inspect.signature(method).parameters.values()
        if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
            return modern
        positional = [parameter for parameter in parameters if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        return modern if len(positional) >= len(modern) else modern[:legacy_count]

    async def _reason_person(self, space_id: str, person_id: str) -> None:
        # If Extract updates the same Person while an LLM call is in flight,
        # the optimistic watermark check fails and this loop recomputes from the
        # latest snapshot without taking a lock.
        while not self._stopping:
            context = self.store.person_context(space_id, person_id)
            if not context:
                return
            person, memories, watermark = context
            if not person.stale:
                return
            inferred, hypotheses = self.store.owner_review_context(space_id, "person", person_id)
            result = await self.queue.submit(
                "reason-person", self.model.reason_person, *self._owner_reason_args(self.model.reason_person, (person, memories, inferred, hypotheses), 2)
            )
            if self.store.apply_person_reasoning(
                space_id,
                person_id,
                watermark,
                result,
                {memory.id for memory in inferred}, {hypothesis.id for hypothesis in hypotheses},
            ):
                return

    async def _reason_coverage(self, space_id: str) -> None:
        """Audit all bounded chunks before a single goal-planning pass."""
        while not self._stopping:
            context = self.store.coverage_context(space_id)
            if not context:
                return
            coverage, memories, hypotheses, pending = context
            if not memories:
                return
            audit = getattr(self.model, "audit_coverage", None)
            if audit is None:
                return
            result = await self.queue.submit(
                "audit-coverage", audit, coverage, memories, hypotheses
            )
            if not self.store.apply_coverage_audit(
                space_id, coverage.source_watermark, coverage.source_cursor_id, result,
                {memory.id for memory in memories}, {boundary.id for boundary in coverage.boundaries},
                {hypothesis.id for hypothesis in hypotheses},
            ):
                continue
            if pending:
                continue
            return

    async def _plan_learning_goals(self, space_id: str) -> None:
        context = self.store.learning_goal_context(space_id)
        if not context:
            return
        coverage, hypotheses, open_goals, closed_goals = context
        planner = getattr(self.model, "plan_learning_goals", None)
        if planner is None:
            return
        result = await self.queue.submit(
            "plan-learning-goals", planner,
            coverage, hypotheses, open_goals, closed_goals,
        )
        self.store.apply_goal_planning(space_id, coverage.revision, result, {goal.id for goal in open_goals} | {goal.id for goal in closed_goals})

    async def _reason_user_model(self, space_id: str) -> None:
        while not self._stopping:
            context = self.store.user_model_context(space_id)
            if not context:
                return
            user_model, memories, watermark = context
            if not user_model.stale:
                return
            evidence = [memory for memory in memories if memory.basis != "inferred"]
            inferred, hypotheses = self.store.owner_review_context(space_id, "user", None)
            result = await self.queue.submit(
                "reason-user-model", self.model.reason_user_model, *self._owner_reason_args(self.model.reason_user_model, (evidence, inferred, hypotheses), 1)
            )
            if self.store.apply_user_model_reasoning(space_id, watermark, result, {hypothesis.id for hypothesis in hypotheses}):
                return

    async def _reason_relationship(
        self, space_id: str, relationship_id: str
    ) -> None:
        while not self._stopping:
            context = self.store.relationship_context(space_id, relationship_id)
            if not context:
                return
            relationship, memories, watermark = context
            if not relationship.stale:
                return
            inferred, hypotheses = self.store.owner_review_context(space_id, "relationship", relationship_id)
            result = await self.queue.submit(
                "reason-relationship",
                self.model.reason_relationship,
                *self._owner_reason_args(self.model.reason_relationship, (relationship, memories, inferred, hypotheses), 2),
            )
            if self.store.apply_relationship_reasoning(
                space_id,
                relationship_id,
                watermark,
                result,
                {memory.id for memory in inferred}, {hypothesis.id for hypothesis in hypotheses},
            ):
                return

    async def query(self, space_id: str, request: QueryRequest) -> QueryResponse:
        context = self.store.read(space_id, request)
        answer = await self.queue.submit(
            "query", self.model.synthesize, request.question, context
        )
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
        return HealthResponse(
            llm_configured=self.model.configured,
            queue=self.queue.status(),
        )
