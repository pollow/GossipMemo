"""OpenAI-compatible chat-completions transport adapter.

`transport.py` defines the narrow `LlmTransport` protocol every reasoner
depends on; this module supplies the one production implementation of it,
`OpenAICompatibleAdapter`. It owns all HTTP details -- retries, the
provider backpressure gate, and the hard context-budget check -- and
nothing about how a particular reasoner shapes its prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from types import TracebackType

import httpx
from pydantic import ValidationError

from .config import Settings
from .context_budget import ContextBudget
from .priority import _call_label, _call_tier
from .transport import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMProtocolError,
    LLMRequestError,
    LlmTransport,
    ProviderGate,
    RetryPolicy,
    _message_content,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleAdapter(AbstractAsyncContextManager["OpenAICompatibleAdapter"]):
    """Adapter for servers exposing an OpenAI-compatible ``/chat/completions``.

    ``base_url`` may be either the configured API root or a complete
    chat-completions endpoint. GossipMemo supplies no provider URL default.
    Supplying an ``httpx.AsyncClient`` is supported for tests and for callers
    that manage connection pooling themselves; when omitted, this adapter
    owns a lazily-created client and closes it in :meth:`aclose`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        headers: Mapping[str, str] | None = None,
        max_retries: int = 5,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        context_budget: ContextBudget | None = None,
        trace_path: Path | None = None,
    ) -> None:
        normalized_base = base_url.strip().rstrip("/")
        if not normalized_base:
            raise ValueError("LLM base_url must not be empty")
        if not model.strip():
            raise ValueError("LLM model must not be empty")
        if not api_key.strip():
            raise ValueError("LLM api_key must not be empty")
        if timeout <= 0:
            raise ValueError("LLM timeout must be greater than zero")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("LLM max_tokens must be greater than zero")
        if max_retries < 0:
            raise ValueError("LLM max_retries must not be negative")
        if retry_base_seconds <= 0:
            raise ValueError("LLM retry_base_seconds must be greater than zero")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError(
                "LLM retry_max_seconds must be at least retry_base_seconds"
            )

        self.base_url = normalized_base
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.context_budget = context_budget or ContextBudget()
        self.trace_path = trace_path
        self._trace_sequence = count(1)
        self._trace_disabled = False
        self._client = client
        self._owns_client = client is None
        self._headers = dict(headers or {})
        self._gate = ProviderGate()

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleAdapter:
        """Build the Adapter from the already-validated global settings."""

        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_base_seconds=settings.llm_retry_base_seconds,
            retry_max_seconds=settings.llm_retry_max_seconds,
            context_budget=ContextBudget(
                settings.llm_context_window_tokens,
                settings.llm_output_reserve_tokens,
                settings.llm_context_safety_tokens,
            ),
            trace_path=settings.llm_trace_path,
        )

    @property
    def configured(self) -> bool:
        return True

    @property
    def gate(self) -> ProviderGate:
        """The process-global provider backpressure gate for this adapter."""

        return self._gate

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def prepare(
        self, messages: Sequence[ChatMessage], *, structured: bool
    ) -> ChatCompletionRequest:
        """Build the request this adapter would send for `messages`.

        The one place request-shaping configuration is applied, so a caller
        can estimate the exact request before deciding to send it.
        """
        return ChatCompletionRequest(
            model=self.model,
            messages=list(messages),
            temperature=self.temperature,
            response_format={"type": "json_object"} if structured else None,
            max_tokens=self.max_tokens,
        )

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.max_retries,
            base_seconds=self.retry_base_seconds,
            max_seconds=self.retry_max_seconds,
        )

    async def complete(self, request: ChatCompletionRequest, *, trace: bool = True) -> str:
        """Send a prebuilt chat-completions request over HTTP.

        Acquires :attr:`gate`, runs the context-budget hard check, POSTs,
        and runs the existing transport retry/backoff (including
        `mark_unavailable_until` on 429/5xx). Returns the raw assistant
        message content. `response_format` is decided by the caller when
        it builds `request`, not here.

        `trace=False` skips this call's `_trace` write entirely. The admin
        playground is the one caller that passes it: it replays a call
        through this same adapter (so it still serializes behind `gate`)
        but writes its own trace record into a reserved sibling directory,
        since it knows the exact path it wrote and redirects straight to
        it. Every other caller leaves this at the default.
        """
        structured = request.response_format is not None
        # Check the exact serialized request before acquiring/sending HTTP.
        # This includes system/messages and response schema JSON.
        self.context_budget.check(self.context_budget.estimate_request(request))
        client = await self._get_client()
        headers = {"Accept": "application/json", **self._headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        tier = _call_tier.get()
        label = _call_label.get() or f"tier{tier}-{'structured' if structured else 'chat'}"
        # Serialize at the single provider request, not the whole reasoner
        # call. Holding the gate across this entire retry loop -- including
        # its `asyncio.sleep` backoff -- is deliberate global backpressure:
        # while the provider is unavailable, no other request goes out
        # either. Semantic-retry loops for malformed structured output live
        # outside this method and never touch the gate.
        await self._gate.acquire(tier, label)
        try:
            attempt = 0
            while True:
                started = time.perf_counter()
                try:
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        json=request.model_dump(exclude_none=True),
                    )
                except httpx.TransportError as error:
                    if attempt >= self.max_retries:
                        logger.exception(
                            "llm_call_transport_failed",
                            extra={
                                "model": self.model,
                                "attempt": attempt + 1,
                                "duration_ms": round(
                                    (time.perf_counter() - started) * 1000, 2
                                ),
                            },
                        )
                        raise LLMRequestError(f"LLM request failed: {error}") from error
                    delay = self._retry_delay(attempt)
                    self._gate.mark_unavailable_until(
                        asyncio.get_running_loop().time() + delay
                    )
                    logger.warning(
                        "llm_call_retry_scheduled",
                        extra={
                            "model": self.model,
                            "attempt": attempt + 1,
                            "reason": type(error).__name__,
                            "delay_seconds": round(delay, 3),
                        },
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                logger.info(
                    "llm_call_completed",
                    extra={
                        "model": self.model,
                        "status": response.status_code,
                        "attempt": attempt + 1,
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                        "structured": structured,
                    },
                )

                if not response.is_error:
                    break
                if _retryable_status(response.status_code) and attempt < self.max_retries:
                    delay = self._retry_delay(
                        attempt, _retry_after_seconds(response.headers.get("Retry-After"))
                    )
                    self._gate.mark_unavailable_until(
                        asyncio.get_running_loop().time() + delay
                    )
                    logger.warning(
                        "llm_call_retry_scheduled",
                        extra={
                            "model": self.model,
                            "attempt": attempt + 1,
                            "status": response.status_code,
                            "delay_seconds": round(delay, 3),
                        },
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                detail = _response_detail(response)
                raise LLMRequestError(
                    f"LLM request failed with HTTP {response.status_code}: {detail}"
                )
        finally:
            self._gate.release()

        try:
            payload = response.json()
            completion = ChatCompletionResponse.model_validate(payload)
        except (ValueError, ValidationError) as error:
            if trace:
                self._trace(request, label, tier, response.status_code,
                            error="invalid chat-completion response")
            raise LLMProtocolError("LLM returned an invalid chat-completion response") from error
        message = completion.choices[0].message
        content = _message_content(message.content)
        if trace:
            self._trace(request, label, tier, response.status_code, completion=content)
        return content

    def _trace(
        self, request: ChatCompletionRequest, label: str, tier: int, status: int,
        *, completion: str | None = None, error: str | None = None,
    ) -> None:
        """Write one request/response pair to its own trace file, verbatim.

        Off unless `GOSSIPMEMO_LLM_TRACE_PATH` is set. This is the audit
        seam for prompt construction, so nothing here is truncated or
        summarized: what the record holds is exactly what went over the
        wire, plus the reasoner `label` that structured logging alone
        cannot recover.

        One JSON file is written per call, under a per-day subdirectory of
        `trace_path` (a directory), named by local time so an operator can
        pick days the way they experience them:

            <trace_path>/<YYYY-MM-DD>/<HHMMSSmmm>-<label>-<sequence>.json

        `timestamp` inside the record itself stays UTC -- only the
        filename and directory use local time. The file is opened
        exclusively (`O_CREAT | O_EXCL`) and a numeric suffix is added on
        collision, since `self._trace_sequence` resets on process restart
        and does not by itself guarantee a unique name. Failures never
        propagate -- a broken trace must not take a reasoning pass down
        with it. If `trace_path` turns out to be an existing regular file
        (the old single-JSONL-file layout), tracing is disabled for the
        rest of the process instead of crashing.
        """
        if self.trace_path is None or self._trace_disabled:
            return
        if self.trace_path.is_file():
            logger.warning(
                "llm_trace_path_is_file", extra={"path": str(self.trace_path)}
            )
            self._trace_disabled = True
            return
        sequence = next(self._trace_sequence)
        record = build_trace_record(
            sequence=sequence,
            label=label,
            tier=tier,
            model=self.model,
            status=status,
            estimated_tokens=self.context_budget.estimate_request(request),
            request=request,
            completion=completion,
            error=error,
        )
        write_trace_file(self.trace_path, record, label, sequence)

    def _retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        return self.retry_policy.delay(attempt, retry_after)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the owned HTTP client; injected clients remain caller-owned."""

        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await client.aclose()

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> OpenAICompatibleAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def build_trace_record(
    *,
    sequence: int,
    label: str,
    tier: int,
    model: str,
    status: int,
    estimated_tokens: int,
    request: ChatCompletionRequest,
    completion: str | None = None,
    error: str | None = None,
) -> dict:
    """Build one trace record, in the exact shape `_trace` has always written.

    Pulled out of `_trace` so the admin playground (`admin/views/playground.py`)
    can build a replay record with the same shape without constructing a
    second `OpenAICompatibleAdapter` -- it calls the shared adapter with
    `complete(..., trace=False)` and writes its own record via this and
    `write_trace_file` instead, so it knows the exact path it wrote.
    """

    return {
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "tier": tier,
        "model": model,
        "status": status,
        "estimated_tokens": estimated_tokens,
        "request": request.model_dump(exclude_none=True),
        "completion": completion,
        "error": error,
    }


def write_trace_file(directory: Path, record: dict, label: str, sequence: int) -> Path | None:
    """Write `record` as one JSON file under a per-day subdirectory of `directory`.

    Same layout and collision handling `_trace` has always used:

        <directory>/<YYYY-MM-DD>/<HHMMSSmmm>-<label>-<sequence>.json

    (local time for the directory and filename; `record["timestamp"]` is
    whatever the caller put there, normally UTC). Returns the path written,
    or `None` on any failure -- callers must not let a trace-write failure
    take anything else down with it.
    """

    now_local = datetime.now().astimezone()
    day_dir = directory / now_local.strftime("%Y-%m-%d")
    time_prefix = now_local.strftime("%H%M%S") + f"{now_local.microsecond // 1000:03d}"
    safe_label = re.sub(r"[^a-z0-9-]", "-", label.lower())
    payload = json.dumps(record, indent=2, ensure_ascii=False, default=str)
    try:
        day_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{time_prefix}-{safe_label}-{sequence:04d}"
        max_attempts = 10
        for attempt in range(max_attempts):
            name = base_name if attempt == 0 else f"{base_name}-{attempt}"
            target = day_dir / f"{name}.json"
            try:
                fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
            finally:
                pass
            return target
        logger.warning(
            "llm_trace_write_failed",
            extra={"path": str(day_dir), "reason": "exhausted unique name attempts"},
        )
        return None
    except OSError:
        logger.warning("llm_trace_write_failed", extra={"path": str(directory)})
        return None


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        detail = response.text
    else:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or json.dumps(error, ensure_ascii=False)
            else:
                detail = payload.get("message") or json.dumps(payload, ensure_ascii=False)
        else:
            detail = json.dumps(payload, ensure_ascii=False)
    return str(detail).strip()[:1000] or "no response body"


def _retryable_status(status_code: int) -> bool:
    return status_code in {408, 429} or 500 <= status_code <= 599


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)


def create_llm(settings: Settings) -> LlmTransport:
    """Build the model Adapter from validated process settings."""

    return OpenAICompatibleAdapter.from_settings(settings)


__all__ = [
    "OpenAICompatibleAdapter",
    "build_trace_record",
    "create_llm",
    "write_trace_file",
]
