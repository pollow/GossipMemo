"""A process-local, strict FIFO queue for LLM calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .models import QueueStatus

logger = logging.getLogger(__name__)


T = TypeVar("T")
AsyncOperation = Callable[..., Awaitable[T]]


class QueueStoppedError(RuntimeError):
    pass


@dataclass(slots=True)
class _Job(Generic[T]):
    label: str
    operation: AsyncOperation[T]
    args: tuple[Any, ...]
    future: asyncio.Future[T]


class SequentialLLMQueue:
    """Run accepted LLM calls one at a time, in submission order.

    This is deliberately not a durable job system. Message processing state in
    SQLite is the recovery source; restarting the server simply re-enqueues
    unfinished work.
    """

    def __init__(self) -> None:
        self._jobs: asyncio.Queue[_Job[Any] | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False
        self._current_label: str | None = None

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._accepting = True
        self._worker = asyncio.create_task(
            self._run(), name="gossipmemo-llm-queue"
        )
        logger.info("llm_queue_started")

    async def submit(
        self,
        label: str,
        operation: AsyncOperation[T],
        *args: Any,
    ) -> T:
        if not self._accepting:
            raise QueueStoppedError("LLM queue is not running")
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        await self._jobs.put(_Job(label, operation, args, future))
        return await future

    async def stop(self) -> None:
        if not self._worker:
            self._accepting = False
            return
        if self._accepting:
            self._accepting = False
            await self._jobs.put(None)
        await self._worker
        self._worker = None
        logger.info("llm_queue_stopped")

    async def _run(self) -> None:
        while True:
            job = await self._jobs.get()
            try:
                if job is None:
                    return
                self._current_label = job.label
                started = time.perf_counter()
                logger.info("llm_queue_job_started", extra={"operation": job.label, "pending": self._jobs.qsize()})
                try:
                    result = await job.operation(*job.args)
                except Exception as error:
                    logger.exception("llm_queue_job_failed", extra={"operation": job.label, "duration_ms": round((time.perf_counter() - started) * 1000, 2)})
                    if not job.future.done():
                        job.future.set_exception(error)
                else:
                    logger.info("llm_queue_job_completed", extra={"operation": job.label, "duration_ms": round((time.perf_counter() - started) * 1000, 2)})
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    self._current_label = None
            finally:
                self._jobs.task_done()

    def status(self) -> QueueStatus:
        pending = self._jobs.qsize()
        if not self._accepting and pending:
            pending -= 1  # Do not expose the shutdown sentinel as work.
        return QueueStatus(
            pending=max(0, pending),
            running=bool(self._worker and not self._worker.done()),
            current_label=self._current_label,
        )


# A short compatibility spelling for callers created during the first draft.
LLMQueue = SequentialLLMQueue


__all__ = ["LLMQueue", "QueueStoppedError", "SequentialLLMQueue"]
