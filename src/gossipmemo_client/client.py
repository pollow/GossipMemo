"""Small synchronous and asynchronous clients for the GossipMemo HTTP API.

The server intentionally owns the memory model.  This module is only a
transport adapter: it does not try to mirror the server's pydantic models and
returns the JSON objects supplied by the API.  Keeping the client this small
also makes it useful from integrations which do not install the server
package (for example, a Hermes plugin).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any, TypeVar
from urllib.parse import quote

import httpx


Json = dict[str, Any]
_ClientT = TypeVar("_ClientT", httpx.Client, httpx.AsyncClient)


def _jsonable(value: Any) -> Any:
    """Convert common model values into values accepted by ``httpx`` JSON."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json", exclude_none=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _normalise_messages(messages: Any) -> list[Json]:
    """Accept a pydantic model, one mapping, or an iterable of either."""

    if isinstance(messages, Mapping) or callable(getattr(messages, "model_dump", None)):
        items: Iterable[Any] = [messages]
    else:
        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a mapping or iterable of mappings")
        try:
            items = iter(messages)
        except TypeError as exc:
            raise TypeError("messages must be a mapping or iterable of mappings") from exc

    result: list[Json] = []
    for item in items:
        value = _jsonable(item)
        if not isinstance(value, Mapping):
            raise TypeError("each message must be a mapping or model with model_dump()")
        result.append(dict(value))
    if not result:
        raise ValueError("messages must contain at least one message")
    return result


def _response_detail(response: httpx.Response) -> str:
    """Get a useful, bounded error detail from a JSON or text response."""

    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, Mapping):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if detail is not None:
            if isinstance(detail, (Mapping, list)):
                return str(detail)[:2000]
            return str(detail)[:2000]
    if body:
        return str(body)[:2000]
    return "no response body"


class GossipMemoError(RuntimeError):
    """Base error raised when a GossipMemo request cannot be completed.

    ``status_code`` is set for an HTTP error and ``None`` for a transport or
    decoding error.  ``response_body`` is intentionally retained in bounded
    form because it is often the only useful diagnostic when the server is
    deployed separately from the agent.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        method: str | None = None,
        url: str | None = None,
        response_body: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.response_body = response_body
        self.cause = cause
        prefix = f"{method} {url}: " if method and url else ""
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"{prefix}{message}{status}")


class _ClientCommon:
    """Shared validation and URL/header behaviour for both clients."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        space_id: str = "personal",
        *,
        timeout: float | httpx.Timeout = 10.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url is required")
        if not isinstance(space_id, str) or not space_id.strip():
            raise ValueError("space_id is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip() if isinstance(api_key, str) else str(api_key or "")
        self.space_id = space_id.strip()
        self.timeout = timeout
        self._headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            token = self.api_key
            if not token.lower().startswith("bearer "):
                token = f"Bearer {token}"
            self._headers["Authorization"] = token
        if headers:
            self._headers.update({str(key): str(value) for key, value in headers.items()})

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _space_path(self, suffix: str) -> str:
        space = quote(self.space_id, safe="")
        return f"/v1/spaces/{space}/{suffix.lstrip('/')}"

class GossipMemo(_ClientCommon):
    """Synchronous GossipMemo client.

    Parameters are deliberately explicit so an integration cannot silently
    send memories to the wrong space.  A custom ``httpx.Client`` or transport
    can be supplied for connection pooling and tests; owned clients are closed
    by :meth:`close` and custom clients are left open.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        space_id: str = "personal",
        *,
        timeout: float | httpx.Timeout = 10.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(base_url, api_key, space_id, timeout=timeout, headers=headers)
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout, transport=transport)

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        url = self._url(path)
        try:
            response = self._client.request(
                method,
                url,
                headers=self._headers,
                json=_jsonable(payload) if payload is not None else None,
            )
        except (httpx.HTTPError, TimeoutError) as exc:
            raise GossipMemoError(
                f"request failed: {exc}", method=method, url=url, cause=exc
            ) from exc
        if response.is_error:
            raise GossipMemoError(
                _response_detail(response),
                status_code=response.status_code,
                method=method,
                url=url,
                response_body=response.text[:2000],
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GossipMemoError(
                "server returned invalid JSON",
                status_code=response.status_code,
                method=method,
                url=url,
                response_body=response.text[:2000],
                cause=exc,
            ) from exc

    def health(self) -> Any:
        """Return the server health document."""

        return self._request("GET", "/health")

    def ingest(
        self,
        messages: Any = None,
        *,
        content: str | None = None,
        author: Any = None,
        source: Any = None,
        occurred_at: datetime | date | str | None = None,
        idempotency_key: str | None = None,
        extraction_policy: str | None = None,
    ) -> Any:
        """Persist one or more messages and queue background extraction."""

        if messages is not None and not isinstance(messages, str) and any(
            value is not None
            for value in (
                content,
                author,
                source,
                occurred_at,
                idempotency_key,
                extraction_policy,
            )
        ):
            raise ValueError("pass either messages or single-message fields, not both")
        if messages is None:
            if content is None:
                raise TypeError("ingest requires messages or content")
            if author is None:
                raise ValueError("author is required for ingest")
            messages = {
                "content": content,
                "author": author,
                "source": source,
                "occurred_at": occurred_at,
                "idempotency_key": idempotency_key,
                "extraction_policy": extraction_policy,
            }
        elif isinstance(messages, str):
            messages = {
                "content": messages,
                "author": author,
                "source": source,
                "occurred_at": occurred_at,
                "idempotency_key": idempotency_key,
                "extraction_policy": extraction_policy,
            }
        return self._request(
            "POST", self._space_path("ingest"), {"messages": _normalise_messages(messages)}
        )

    def query(
        self,
        question: str | Mapping[str, Any],
        *,
        people: Sequence[str] | None = None,
        include_relationships: bool = True,
        expand_relationships: int = 0,
        include_evidence: bool = True,
        limit: int = 30,
        **extra: Any,
    ) -> Any:
        """Query memories and relationship/person projections."""

        if isinstance(question, Mapping):
            payload = dict(question)
            payload.update(extra)
        else:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question is required")
            payload = {
                "question": question,
                "people": list(people or []),
                "include_relationships": include_relationships,
                "expand_relationships": expand_relationships,
                "include_evidence": include_evidence,
                "limit": limit,
                **extra,
            }
        return self._request("POST", self._space_path("query"), payload)

    def add_memory(
        self,
        content: str | Mapping[str, Any],
        *,
        kind: str = "fact",
        people: Sequence[Mapping[str, Any] | Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        **extra: Any,
    ) -> Any:
        """Create a manual memory in the configured space."""

        if isinstance(content, Mapping):
            payload = dict(content)
            payload.update(extra)
        else:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content is required")
            payload = {
                "content": content,
                "kind": kind,
                "people": list(people or []),
                "valid_from": valid_from,
                "valid_to": valid_to,
                **extra,
            }
        return self._request("POST", self._space_path("memories"), payload)

    def retract(self, memory_id: str, *, reason: str | None = None) -> Any:
        """Retract an active memory while retaining its provenance."""

        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id is required")
        payload = {"reason": reason} if reason is not None else {}
        identifier = quote(memory_id, safe="")
        return self._request(
            "POST", self._space_path(f"memories/{identifier}/retract"), payload
        )

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        kind: str | None = None,
        reason: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> Any:
        """Replace a memory while preserving its correction history."""

        if not memory_id.strip() or not content.strip():
            raise ValueError("memory_id and content are required")
        payload = {
            "content": content,
            "kind": kind,
            "reason": reason,
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
        identifier = quote(memory_id, safe="")
        return self._request(
            "POST", self._space_path(f"memories/{identifier}/supersede"), payload
        )

    def person_dossier(self, person_id: str) -> Any:
        """Read a person projection without synthesis."""

        if not person_id:
            raise ValueError("person_id is required")
        return self._request(
            "GET", self._space_path(f"people/{quote(person_id, safe='')}"))

    def relationship_dossier(self, relationship_id: str) -> Any:
        """Read a relationship projection without synthesis."""

        if not relationship_id:
            raise ValueError("relationship_id is required")
        return self._request(
            "GET",
            self._space_path(f"relationships/{quote(relationship_id, safe='')}"),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GossipMemo":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncGossipMemo(_ClientCommon):
    """Asynchronous counterpart to :class:`GossipMemo`."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        space_id: str = "personal",
        *,
        timeout: float | httpx.Timeout = 10.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(base_url, api_key, space_id, timeout=timeout, headers=headers)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, transport=transport)

    async def _request(self, method: str, path: str, payload: Any = None) -> Any:
        url = self._url(path)
        try:
            response = await self._client.request(
                method,
                url,
                headers=self._headers,
                json=_jsonable(payload) if payload is not None else None,
            )
        except (httpx.HTTPError, TimeoutError) as exc:
            raise GossipMemoError(
                f"request failed: {exc}", method=method, url=url, cause=exc
            ) from exc
        if response.is_error:
            raise GossipMemoError(
                _response_detail(response),
                status_code=response.status_code,
                method=method,
                url=url,
                response_body=response.text[:2000],
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GossipMemoError(
                "server returned invalid JSON",
                status_code=response.status_code,
                method=method,
                url=url,
                response_body=response.text[:2000],
                cause=exc,
            ) from exc

    async def health(self) -> Any:
        return await self._request("GET", "/health")

    async def ingest(
        self,
        messages: Any = None,
        *,
        content: str | None = None,
        author: Any = None,
        source: Any = None,
        occurred_at: datetime | date | str | None = None,
        idempotency_key: str | None = None,
        extraction_policy: str | None = None,
    ) -> Any:
        if messages is not None and not isinstance(messages, str) and any(
            value is not None
            for value in (
                content,
                author,
                source,
                occurred_at,
                idempotency_key,
                extraction_policy,
            )
        ):
            raise ValueError("pass either messages or single-message fields, not both")
        if messages is None:
            if content is None:
                raise TypeError("ingest requires messages or content")
            if author is None:
                raise ValueError("author is required for ingest")
            messages = {
                "content": content,
                "author": author,
                "source": source,
                "occurred_at": occurred_at,
                "idempotency_key": idempotency_key,
                "extraction_policy": extraction_policy,
            }
        elif isinstance(messages, str):
            messages = {
                "content": messages,
                "author": author,
                "source": source,
                "occurred_at": occurred_at,
                "idempotency_key": idempotency_key,
                "extraction_policy": extraction_policy,
            }
        return await self._request(
            "POST", self._space_path("ingest"), {"messages": _normalise_messages(messages)}
        )

    async def query(
        self,
        question: str | Mapping[str, Any],
        *,
        people: Sequence[str] | None = None,
        include_relationships: bool = True,
        expand_relationships: int = 0,
        include_evidence: bool = True,
        limit: int = 30,
        **extra: Any,
    ) -> Any:
        if isinstance(question, Mapping):
            payload = dict(question)
            payload.update(extra)
        else:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question is required")
            payload = {
                "question": question,
                "people": list(people or []),
                "include_relationships": include_relationships,
                "expand_relationships": expand_relationships,
                "include_evidence": include_evidence,
                "limit": limit,
                **extra,
            }
        return await self._request("POST", self._space_path("query"), payload)

    async def add_memory(
        self,
        content: str | Mapping[str, Any],
        *,
        kind: str = "fact",
        people: Sequence[Mapping[str, Any] | Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        **extra: Any,
    ) -> Any:
        if isinstance(content, Mapping):
            payload = dict(content)
            payload.update(extra)
        else:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content is required")
            payload = {
                "content": content,
                "kind": kind,
                "people": list(people or []),
                "valid_from": valid_from,
                "valid_to": valid_to,
                **extra,
            }
        return await self._request("POST", self._space_path("memories"), payload)

    async def retract(self, memory_id: str, *, reason: str | None = None) -> Any:
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id is required")
        payload = {"reason": reason} if reason is not None else {}
        identifier = quote(memory_id, safe="")
        return await self._request(
            "POST", self._space_path(f"memories/{identifier}/retract"), payload
        )

    async def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        kind: str | None = None,
        reason: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> Any:
        if not memory_id.strip() or not content.strip():
            raise ValueError("memory_id and content are required")
        payload = {
            "content": content,
            "kind": kind,
            "reason": reason,
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
        identifier = quote(memory_id, safe="")
        return await self._request(
            "POST", self._space_path(f"memories/{identifier}/supersede"), payload
        )

    async def person_dossier(self, person_id: str) -> Any:
        if not person_id:
            raise ValueError("person_id is required")
        return await self._request(
            "GET", self._space_path(f"people/{quote(person_id, safe='')}"))

    async def relationship_dossier(self, relationship_id: str) -> Any:
        if not relationship_id:
            raise ValueError("relationship_id is required")
        return await self._request(
            "GET",
            self._space_path(f"relationships/{quote(relationship_id, safe='')}"),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncGossipMemo":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


__all__ = [
    "AsyncGossipMemo",
    "GossipMemo",
    "GossipMemoError",
]
