"""LLM interface and an OpenAI-compatible chat-completions adapter.

The rest of GossipMemo depends on the small :class:`LlmModel` interface in
this module.  The concrete adapter is deliberately responsible for all HTTP
details and for validating model output against the domain Pydantic models;
callers never need to parse a chat-completions response themselves.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Literal, Protocol, TypeVar, cast, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .models import (
    ExtractionResult,
    MemoryView,
    ModelMessage,
    PersonReasoningResult,
    PersonView,
    QueryContext,
    RelationshipReasoningResult,
    RelationshipView,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    PERSON_REASONING_SYSTEM_PROMPT,
    QUERY_SYNTHESIS_SYSTEM_PROMPT,
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    extraction_prompt,
    person_reasoning_prompt,
    query_synthesis_prompt,
    relationship_reasoning_prompt,
    schema_instruction,
)


@runtime_checkable
class LlmModel(Protocol):
    """The application-facing asynchronous LLM seam.

    Implementations may be deterministic fakes, a hosted model adapter, or
    :class:`UnavailableLlm`.  ``configured`` is intentionally separate from
    an invocation so health endpoints can report configuration without making
    a network request.
    """

    @property
    def configured(self) -> bool: ...

    async def extract(self, message: ModelMessage) -> ExtractionResult: ...

    async def reason_person(
        self, person: PersonView, memories: Sequence[MemoryView]
    ) -> PersonReasoningResult: ...

    async def reason_relationship(
        self, relationship: RelationshipView, memories: Sequence[MemoryView]
    ) -> RelationshipReasoningResult: ...

    async def synthesize(self, question: str, context: QueryContext) -> str: ...


LLMModel = LlmModel


class LLMError(RuntimeError):
    """Base class for model configuration, transport, and output failures."""


class ModelUnavailableError(LLMError):
    """Raised when no usable model configuration has been supplied."""


class LLMRequestError(LLMError):
    """Raised when the chat-completions HTTP request fails."""


class LLMProtocolError(LLMError):
    """Raised when a provider response is not a valid chat completion."""


class LLMOutputError(LLMError):
    """Raised when model content cannot validate as the requested result."""


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


class UnavailableLlm:
    """Explicit adapter used when model settings are absent.

    Returning this adapter from application setup keeps configuration visible
    in health checks while ensuring a queued extraction/reasoning/query job
    fails with a clear, actionable error rather than an accidental network
    call or an obscure ``None`` dereference.
    """

    def __init__(self, reason: str = "LLM is not configured") -> None:
        self.reason = reason.strip() or "LLM is not configured"

    @property
    def configured(self) -> bool:
        return False

    def _error(self) -> ModelUnavailableError:
        return ModelUnavailableError(self.reason)

    async def extract(self, message: ModelMessage) -> ExtractionResult:
        del message
        raise self._error()

    async def reason_person(
        self, person: PersonView, memories: Sequence[MemoryView]
    ) -> PersonReasoningResult:
        del person, memories
        raise self._error()

    async def reason_relationship(
        self, relationship: RelationshipView, memories: Sequence[MemoryView]
    ) -> RelationshipReasoningResult:
        del relationship, memories
        raise self._error()

    async def synthesize(self, question: str, context: QueryContext) -> str:
        del question, context
        raise self._error()

class OpenAICompatibleAdapter(AbstractAsyncContextManager["OpenAICompatibleAdapter"]):
    """Adapter for OpenAI and servers exposing ``/chat/completions``.

    ``base_url`` may be either the API root (for example
    ``https://api.openai.com/v1``) or a complete chat-completions endpoint.
    Supplying an ``httpx.AsyncClient`` is supported for tests and for callers
    that manage connection pooling themselves; when omitted, this adapter
    owns a lazily-created client and closes it in :meth:`aclose`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        normalized_base = base_url.strip().rstrip("/")
        if not normalized_base:
            raise ValueError("LLM base_url must not be empty")
        if not model.strip():
            raise ValueError("LLM model must not be empty")
        if timeout <= 0:
            raise ValueError("LLM timeout must be greater than zero")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("LLM max_tokens must be greater than zero")

        self.base_url = normalized_base
        self.api_key = (api_key or "").strip()
        self.model = model.strip()
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        self._owns_client = client is None
        self._headers = dict(headers or {})

    @classmethod
    def from_settings(cls, settings: Settings) -> LlmModel:
        """Build an adapter or explicit unavailable implementation.

        A model name is required for every provider.  Hosted OpenAI's default
        endpoint additionally requires an API key; local/custom endpoints may
        intentionally operate without one.  This keeps Ollama-style local
        deployments possible while making the default unconfigured state
        explicit.
        """

        base_url = (settings.llm_base_url or "").strip().rstrip("/")
        missing: list[str] = []
        if not base_url:
            missing.append("GOSSIPMEMO_LLM_BASE_URL")
        if not (settings.llm_model or "").strip():
            missing.append("GOSSIPMEMO_LLM_MODEL")
        is_openai_default = base_url in {
            "https://api.openai.com/v1",
            "https://api.openai.com",
        }
        if is_openai_default and not (settings.llm_api_key or "").strip():
            missing.append("GOSSIPMEMO_LLM_API_KEY")
        if missing:
            return UnavailableLlm(
                "LLM is not configured; set " + ", ".join(missing)
            )
        return cls(
            base_url=base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )

    @property
    def configured(self) -> bool:
        if not self.base_url or not self.model:
            return False
        if self.base_url in {
            "https://api.openai.com/v1",
            "https://api.openai.com",
        }:
            return bool(self.api_key)
        return True

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    async def extract(self, message: ModelMessage) -> ExtractionResult:
        content = await self._structured_call(
            EXTRACTION_SYSTEM_PROMPT,
            extraction_prompt(message),
            ExtractionResult,
        )
        return cast(ExtractionResult, content)

    async def reason_person(
        self, person: PersonView, memories: Sequence[MemoryView]
    ) -> PersonReasoningResult:
        content = await self._structured_call(
            PERSON_REASONING_SYSTEM_PROMPT,
            person_reasoning_prompt(person, list(memories)),
            PersonReasoningResult,
        )
        return cast(PersonReasoningResult, content)

    async def reason_relationship(
        self, relationship: RelationshipView, memories: Sequence[MemoryView]
    ) -> RelationshipReasoningResult:
        content = await self._structured_call(
            RELATIONSHIP_REASONING_SYSTEM_PROMPT,
            relationship_reasoning_prompt(relationship, list(memories)),
            RelationshipReasoningResult,
        )
        return cast(RelationshipReasoningResult, content)

    async def synthesize(self, question: str, context: QueryContext) -> str:
        if not question.strip():
            raise ValueError("query question must not be empty")
        content = await self._chat(
            QUERY_SYNTHESIS_SYSTEM_PROMPT,
            query_synthesis_prompt(question, context),
            structured=False,
        )
        return content.strip()

    async def _structured_call(
        self,
        system_prompt: str,
        user_prompt: str,
        result_type: type[BaseModel],
    ) -> BaseModel:
        content = await self._chat(
            system_prompt + "\n\n" + schema_instruction(result_type),
            user_prompt,
            structured=True,
        )
        return _parse_model_output(content, result_type)

    async def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
    ) -> str:
        if not self.configured:
            raise ModelUnavailableError(
                "OpenAI-compatible LLM is not configured; provide a model and API key"
            )
        request = ChatCompletionRequest(
            model=self.model,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"} if structured else None,
            max_tokens=self.max_tokens,
        )
        client = await self._get_client()
        headers = {"Accept": "application/json", **self._headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        try:
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=request.model_dump(exclude_none=True),
            )
        except httpx.HTTPError as error:
            raise LLMRequestError(f"LLM request failed: {error}") from error

        if response.is_error:
            detail = _response_detail(response)
            raise LLMRequestError(
                f"LLM request failed with HTTP {response.status_code}: {detail}"
            )
        try:
            payload = response.json()
            completion = ChatCompletionResponse.model_validate(payload)
        except (ValueError, ValidationError) as error:
            raise LLMProtocolError("LLM returned an invalid chat-completion response") from error
        message = completion.choices[0].message
        return _message_content(message.content)

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


def create_llm(settings: Settings) -> LlmModel:
    """Return the configured adapter, or an explicit unavailable adapter."""

    return OpenAICompatibleAdapter.from_settings(settings)


__all__ = [
    "LLMModel",
    "LLMError",
    "LLMOutputError",
    "LLMProtocolError",
    "LLMRequestError",
    "LlmModel",
    "ModelUnavailableError",
    "OpenAICompatibleAdapter",
    "UnavailableLlm",
    "create_llm",
]
