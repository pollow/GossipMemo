from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any

from .llm import LLMModel
from .models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ManualMemoryRequest,
    QueryRequest,
    QueryResponse,
    SupersedeRequest,
)
from .queue import SequentialLLMQueue
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
    ) -> None:
        self.store = store
        self.model = model
        self.queue = queue or SequentialLLMQueue()
        self.extraction_batch_size = extraction_batch_size
        self.extraction_batch_timeout_seconds = extraction_batch_timeout_seconds
        self._tasks: set[asyncio.Task[Any]] = set()
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._scheduled: set[tuple[str, str, str]] = set()
        self._stopping = False

    async def start(self) -> None:
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
        people, relationships = self.store.stale_entities()
        for space_id, person_id in people:
            self._schedule_person_reason(space_id, person_id)
        for space_id, relationship_id in relationships:
            self._schedule_relationship_reason(space_id, relationship_id)

    async def stop(self) -> None:
        self._stopping = True
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
        message_ids = self.store.record_messages(space_id, request.messages)
        self._drain_extraction_batches(space_id)
        return IngestResponse(message_ids=message_ids)

    def _drain_extraction_batches(self, space_id: str) -> None:
        pending = self.store.unbatched_messages(space_id)
        while len(pending) >= self.extraction_batch_size:
            batch_id = self.store.create_extraction_batch(
                space_id,
                [message_id for message_id, _ in pending[: self.extraction_batch_size]],
            )
            if batch_id:
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
        self._spawn(
            ("extract", space_id, batch_id),
            self._extract(space_id, batch_id),
        )

    async def _extract(self, space_id: str, batch_id: str) -> None:
        messages = self.store.load_batch(space_id, batch_id)
        if not messages:
            return
        self.store.mark_extraction_attempt(space_id, batch_id)
        try:
            result = await self.queue.submit("extract", self.model.extract, messages)
            people, relationships = self.store.apply_extraction(
                space_id, batch_id, result
            )
        except Exception as error:
            self.store.fail_extraction(space_id, batch_id, str(error))
            logger.exception("extract failed for %s", batch_id)
            return
        if self._stopping:
            return
        for person_id in people:
            self._schedule_person_reason(space_id, person_id)
        for relationship_id in relationships:
            self._schedule_relationship_reason(space_id, relationship_id)

    def _schedule_person_reason(self, space_id: str, person_id: str) -> None:
        self._spawn(
            ("person", space_id, person_id),
            self._reason_person(space_id, person_id),
        )

    async def _reason_person(self, space_id: str, person_id: str) -> None:
        # If Extract updates the same Person while an LLM call is in flight,
        # the optimistic revision check fails and this loop recomputes from the
        # latest snapshot without taking a lock.
        while not self._stopping:
            context = self.store.person_context(space_id, person_id)
            if not context:
                return
            person, memories = context
            if not person.stale:
                return
            result = await self.queue.submit(
                "reason-person", self.model.reason_person, person, memories
            )
            if self.store.apply_person_reasoning(
                space_id,
                person_id,
                person.memory_revision,
                result,
            ):
                return

    def _schedule_relationship_reason(
        self, space_id: str, relationship_id: str
    ) -> None:
        self._spawn(
            ("relationship", space_id, relationship_id),
            self._reason_relationship(space_id, relationship_id),
        )

    async def _reason_relationship(
        self, space_id: str, relationship_id: str
    ) -> None:
        while not self._stopping:
            context = self.store.relationship_context(space_id, relationship_id)
            if not context:
                return
            relationship, memories = context
            if not relationship.stale:
                return
            result = await self.queue.submit(
                "reason-relationship",
                self.model.reason_relationship,
                relationship,
                memories,
            )
            if self.store.apply_relationship_reasoning(
                space_id,
                relationship_id,
                relationship.memory_revision,
                result,
            ):
                return

    async def query(self, space_id: str, request: QueryRequest) -> QueryResponse:
        context = self.store.read(space_id, request)
        answer = await self.queue.submit(
            "query", self.model.synthesize, request.question, context
        )
        return QueryResponse(answer=answer, **context.model_dump())

    def add_memory(self, space_id: str, request: ManualMemoryRequest) -> str:
        memory_id = self.store.add_manual_memory(space_id, request)
        self._schedule_all_stale()
        return memory_id

    def retract_memory(
        self, space_id: str, memory_id: str, reason: str | None = None
    ) -> bool:
        changed = self.store.retract_memory(space_id, memory_id, reason)
        if changed:
            self._schedule_all_stale()
        return changed

    def supersede_memory(
        self, space_id: str, memory_id: str, request: SupersedeRequest
    ) -> str | None:
        replacement_id = self.store.supersede_memory(space_id, memory_id, request)
        if replacement_id:
            self._schedule_all_stale()
        return replacement_id

    def _schedule_all_stale(self) -> None:
        if self._stopping:
            return
        people, relationships = self.store.stale_entities()
        for space_id, person_id in people:
            self._schedule_person_reason(space_id, person_id)
        for space_id, relationship_id in relationships:
            self._schedule_relationship_reason(space_id, relationship_id)

    def health(self) -> HealthResponse:
        return HealthResponse(
            llm_configured=self.model.configured,
            queue=self.queue.status(),
        )
