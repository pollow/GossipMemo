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
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType

import httpx
from pydantic import ValidationError

from .config import Settings
from .context_budget import ContextBudget
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
from .priority import _call_label, _call_tier

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
        extraction_policy: str = "balanced",
        user_name: str = "CurrentUser",
        max_retries: int = 5,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        context_budget: ContextBudget | None = None,
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
        if extraction_policy not in {"conservative", "balanced", "comprehensive"}:
            raise ValueError(
                "extraction_policy must be conservative, balanced, or comprehensive"
            )
        if not user_name.strip():
            raise ValueError("LLM user_name must not be empty")
        if max_retries < 0:
            raise ValueError("LLM max_retries must not be negative")
        if retry_base_seconds <= 0:
            raise ValueError(
                "LLM retry_base_seconds must be greater than zero")
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
        self.extraction_policy = extraction_policy
        self.user_name = user_name.strip()
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.context_budget = context_budget or ContextBudget()
        self._client = client
        self._owns_client = client is None
        self._headers = dict(headers or {})
        self._gate = ProviderGate()

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleAdapter":
        """Build the Adapter from the already-validated global settings."""

        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            extraction_policy=settings.extraction_policy,
            user_name=settings.user_name,
            max_retries=settings.llm_max_retries,
            retry_base_seconds=settings.llm_retry_base_seconds,
            retry_max_seconds=settings.llm_retry_max_seconds,
            context_budget=ContextBudget(
                settings.llm_context_window_tokens,
                settings.llm_output_reserve_tokens,
                settings.llm_context_safety_tokens,
            ),
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

    async def complete(self, request: ChatCompletionRequest) -> str:
        """Send a prebuilt chat-completions request over HTTP.

        Acquires :attr:`gate`, runs the context-budget hard check, POSTs,
        and runs the existing transport retry/backoff (including
        `mark_unavailable_until` on 429/5xx). Returns the raw assistant
        message content. `response_format` is decided by the caller when
        it builds `request`, not here.
        """
        structured = request.response_format is not None
        # Check the exact serialized request before acquiring/sending HTTP.
        # This includes system/messages and response schema JSON.
        self.context_budget.check(
            self.context_budget.estimate_request(request))
        client = await self._get_client()
        headers = {"Accept": "application/json", **self._headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        tier = _call_tier.get()
        label = _call_label.get(
        ) or f"tier{tier}-{'structured' if structured else 'chat'}"
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
                        raise LLMRequestError(
                            f"LLM request failed: {error}") from error
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
                        attempt, _retry_after_seconds(
                            response.headers.get("Retry-After"))
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
            raise LLMProtocolError(
                "LLM returned an invalid chat-completion response") from error
        message = completion.choices[0].message
        return _message_content(message.content)

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

    async def __aenter__(self) -> "OpenAICompatibleAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        detail = response.text
    else:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or json.dumps(
                    error, ensure_ascii=False)
            else:
                detail = payload.get("message") or json.dumps(
                    payload, ensure_ascii=False)
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
    "create_llm",
]
