"""Provider transport: one chat-completions request, and nothing above it.

This module is a leaf on purpose. `llm.py` carries only the one production
`LlmTransport` implementation, `OpenAICompatibleAdapter`; every reasoner's
prompt assembly lives in `reasoners/` and `query.py` instead, so a
reasoner that also needed `ChatMessage` or `structured()` back from
`llm.py` would close a loop. Everything a reasoner legitimately needs to
talk to a provider lives here, and nothing here imports a reasoner, a
prompt, or a store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .context_budget import ContextBudget
from .priority import (
    TIER_BACKGROUND,
    TIER_FOREGROUND,
    TIER_FRESHNESS,
    llm_call_tier,
)

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for model configuration, transport, and output failures."""


class LLMRequestError(LLMError):
    """Raised when the chat-completions HTTP request fails."""


class LLMProtocolError(LLMError):
    """Raised when a provider response is not a valid chat completion."""


class LLMOutputError(LLMError):
    """Raised when model content cannot validate as the requested result."""


class ProviderGate:
    """Global single-permit priority gate for outbound provider requests.

    The choke point is a single provider HTTP request, not a whole reasoner
    call, so the gate is held per request. Callers `acquire()` immediately before the
    transport-retry loop and `release()` once it returns (success or final
    failure) -- holding the permit across `asyncio.sleep` backoff is
    deliberate, so a 429/5xx/transport backoff blocks every other caller,
    too. Semantic-retry loops (malformed structured output) live outside
    `_chat_messages` and never acquire this gate.

    Concurrency is hardcoded to 1 -- there is intentionally no config knob.
    Priority is strict FIFO within a tier: no aging, no quota. During a large
    backfill, extraction is meant to fully drain before induction runs.
    """

    def __init__(self) -> None:
        self._locked = False
        self._waiters: dict[int, deque[tuple[asyncio.Future[None], str]]] = {
            TIER_FOREGROUND: deque(), TIER_FRESHNESS: deque(), TIER_BACKGROUND: deque(),
        }
        self._unavailable_until: float | None = None
        self._current_tier: int | None = None
        self._current_label: str | None = None

    @property
    def waiting(self) -> int:
        return sum(len(queue) for queue in self._waiters.values())

    @property
    def in_flight(self) -> bool:
        return self._locked

    @property
    def current_label(self) -> str | None:
        return self._current_label

    @property
    def current_tier(self) -> int | None:
        return self._current_tier

    async def acquire(self, tier: int, label: str) -> None:
        await self._wait_out_shared_backoff()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._waiters[tier].append((future, label))
        self._dispatch()
        try:
            await future
        except asyncio.CancelledError:
            try:
                self._waiters[tier].remove((future, label))
            except ValueError:
                # Already granted the permit; hand it to the next waiter.
                if future.done() and not future.cancelled():
                    self.release()
            raise

    def release(self) -> None:
        self._current_tier = None
        self._current_label = None
        self._locked = False
        self._dispatch()

    def mark_unavailable_until(self, deadline: float) -> None:
        """Publish a shared Retry-After deadline so every caller waits it out."""

        if self._unavailable_until is None or deadline > self._unavailable_until:
            self._unavailable_until = deadline

    def _dispatch(self) -> None:
        if self._locked:
            return
        for tier in (TIER_FOREGROUND, TIER_FRESHNESS, TIER_BACKGROUND):
            queue = self._waiters[tier]
            if not queue:
                continue
            future, label = queue.popleft()
            self._locked = True
            self._current_tier = tier
            self._current_label = label
            if not future.done():
                future.set_result(None)
            return

    async def _wait_out_shared_backoff(self) -> None:
        while self._unavailable_until is not None:
            loop = asyncio.get_running_loop()
            delay = self._unavailable_until - loop.time()
            if delay <= 0:
                self._unavailable_until = None
                return
            await asyncio.sleep(delay)


class ChatMessage(BaseModel):
    """Minimal OpenAI-compatible chat message used in request payloads."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    """Pydantic representation of the request sent to a compatible server."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class ChatCompletionMessage(BaseModel):
    """Relevant subset of an OpenAI-compatible response message."""

    model_config = ConfigDict(extra="ignore")

    role: str = "assistant"
    content: str | list[Any] | None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: ChatCompletionMessage


class ChatCompletionResponse(BaseModel):
    """Relevant subset of a chat-completions response."""

    model_config = ConfigDict(extra="ignore")

    choices: list[ChatCompletionChoice] = Field(min_length=1)


@dataclass(frozen=True)
class RetryPolicy:
    """How long to wait before retrying, and how many times to bother.

    One policy object serves both retry loops: the transport's own (inside
    `complete`, which may also have a provider `Retry-After` to respect)
    and the semantic one in `structured` (malformed output, no hint to
    respect). Sharing it keeps a single backoff formula.
    """

    attempts: int
    base_seconds: float
    max_seconds: float

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        exponential = min(self.max_seconds, self.base_seconds * (2**attempt))
        jittered = random.uniform(exponential / 2, exponential)
        if retry_after is None:
            return jittered
        return max(jittered, min(retry_after, self.max_seconds))


@runtime_checkable
class LlmTransport(Protocol):
    """What a reasoner needs from the provider, and nothing it doesn't.

    `prepare` is what keeps this protocol honest: request-shaping config
    (model name, temperature, max_tokens) belongs to the transport, so
    callers hand over messages and get back the exact request that
    `complete` would send. Chunking can then estimate that request against
    `context_budget` with no drift between what was measured and what goes
    out, and `structured` needs no access to adapter internals.

    Every member is about moving one request to a provider and back:
    whether a provider is configured at all, the shared `gate` that
    serializes outbound requests, the `context_budget` a prompt must fit,
    the `retry_policy` both retry loops share, and `prepare`/`complete`/
    `aclose`. Prompt-facing server configuration is not here -- it reaches
    reasoners through `reasoners.ReasoningSettings` instead -- so a
    transport double only has to be a transport.
    """

    @property
    def configured(self) -> bool: ...

    @property
    def gate(self) -> ProviderGate: ...

    @property
    def context_budget(self) -> ContextBudget: ...

    @property
    def retry_policy(self) -> RetryPolicy: ...

    def prepare(
        self, messages: Sequence[ChatMessage], *, structured: bool
    ) -> ChatCompletionRequest: ...

    async def complete(self, request: ChatCompletionRequest, *, trace: bool = True) -> str: ...

    async def aclose(self) -> None: ...


def _message_content(content: str | list[Any] | None) -> str:
    if isinstance(content, str):
        if not content.strip():
            raise LLMProtocolError("LLM returned an empty assistant message")
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        result = "".join(text_parts)
        if result.strip():
            return result
    raise LLMProtocolError("LLM response did not contain assistant text")


_ResultT = TypeVar("_ResultT", bound=BaseModel)


def _parse_model_output(content: str, result_type: type[_ResultT]) -> _ResultT:
    """Parse JSON output, accepting the common fenced-JSON variant."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json\n"):
            text = text[5:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMOutputError("LLM structured output was not valid JSON") from error
    try:
        return result_type.model_validate(value)
    except ValidationError as error:
        raise LLMOutputError(
            f"LLM structured output did not match {result_type.__name__}"
        ) from error


async def structured(
    transport: LlmTransport,
    messages: list[ChatMessage],
    result_type: type[_ResultT],
    *,
    tier: int,
    label: str | None,
) -> tuple[str, _ResultT]:
    """Send a structured-output request over `transport`, retrying malformed output.

    This is policy layered over `LlmTransport.complete`, not part of the
    protocol: it builds the JSON-mode request, marks `tier`/`label` on the
    tier/label contextvars for `complete`'s gate acquisition, and retries
    `LLMOutputError` (but not transport-level failures, which `complete`
    already retries internally) with the same jittered backoff
    `OpenAICompatibleAdapter` has always used for malformed output.

    Everything it needs comes through the protocol: `prepare` shapes the
    request from the transport's own configuration, `retry_policy` supplies
    the attempt count and backoff. Nothing here reaches into a concrete
    adapter, so any `LlmTransport` -- including a test double -- works.
    """
    request = transport.prepare(messages, structured=True)
    policy = transport.retry_policy
    attempt = 0
    while True:
        with llm_call_tier(tier, label):
            content = await transport.complete(request)
        try:
            return content, _parse_model_output(content, result_type)
        except LLMOutputError:
            if attempt >= policy.attempts:
                raise
            delay = policy.delay(attempt)
            logger.warning(
                "llm_output_retry_scheduled",
                extra={
                    "model": request.model,
                    "attempt": attempt + 1,
                    "result_type": result_type.__name__,
                    "delay_seconds": round(delay, 3),
                },
            )
            await asyncio.sleep(delay)
            attempt += 1


__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "LLMError",
    "LLMOutputError",
    "LLMProtocolError",
    "LLMRequestError",
    "LlmTransport",
    "ProviderGate",
    "RetryPolicy",
    "structured",
]
