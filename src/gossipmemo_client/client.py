"""Small synchronous and asynchronous clients for the GossipMemo HTTP API.

The server intentionally owns the memory model.  This module is only a
transport adapter: it does not try to mirror the server's pydantic models and
returns the JSON objects supplied by the API.  Keeping the client this small
also makes it useful from integrations which do not install the server
package (for example, a Hermes plugin).

The two clients differ only in how a request is awaited, so every request is
built once on :class:`_ClientCommon` as a ``(method, path, payload)`` triple
and each public method is a one-line adapter over its own ``_request``.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any
from urllib.parse import quote, urlencode

import httpx

Json = dict[str, Any]
#: A prepared request: HTTP method, path (with query string), and JSON payload.
Call = tuple[str, str, Any]

_TURN_AUTHORS = ("user", "assistant")


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


def _normalise_turn_message(
    message: Any,
    *,
    source: Any = None,
    idempotency_key: str | None = None,
) -> Json:
    """Normalize one message of the batch ``turn`` writes."""
    if isinstance(message, str):
        value: Any = {"content": message}
    else:
        value = _jsonable(message)
    if not isinstance(value, Mapping):
        raise TypeError("message must be a string, mapping, or model")
    result = dict(value)
    author = result.get("author") or "user"
    if author not in _TURN_AUTHORS:
        raise ValueError("message author must be user or assistant")
    result["author"] = author
    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("message content is required")
    result["content"] = content.strip()
    if source is not None:
        result["source"] = _jsonable(source)
    elif result.get("source") is None:
        result.pop("source", None)
    if result.get("source") is not None and not isinstance(result["source"], Mapping):
        raise TypeError("source must be a mapping or model")
    key = idempotency_key if idempotency_key is not None else result.get("idempotency_key")
    if key is None or not str(key).strip():
        key = uuid.uuid4().hex
    result["idempotency_key"] = str(key).strip()
    return result


def _normalise_turn_messages(
    message: Any,
    *,
    source: Any = None,
    idempotency_key: str | None = None,
) -> list[Json]:
    """Accept one message or an iterable of them for ``prepare_turn``."""
    if isinstance(message, str) or isinstance(message, Mapping) or callable(
        getattr(message, "model_dump", None)
    ):
        items: Iterable[Any] = [message]
    else:
        try:
            items = list(message)
        except TypeError as exc:
            raise TypeError(
                "message must be a string, mapping, model, or iterable of those"
            ) from exc
        if not items:
            raise ValueError("messages must contain at least one message")
    return [
        _normalise_turn_message(item, source=source, idempotency_key=idempotency_key)
        for item in items
    ]


def _validate_goals(goals: int | None) -> None:
    """Reject a negative or non-integer learning-goal count before the round trip.

    `None` means "let the server draw its usual random three to five", so a
    negative count is a caller bug rather than a way to ask for that.
    """

    if goals is None:
        return
    if not isinstance(goals, int) or isinstance(goals, bool) or goals < 0:
        raise ValueError("goals must be a non-negative integer or None")


def _goal_params(goals: int | None, goal_seed: str | None) -> list[tuple[str, str]]:
    """Query parameters for the learning-goal knobs; empty when both are defaults."""

    _validate_goals(goals)
    params: list[tuple[str, str]] = []
    if goals is not None:
        params.append(("goals", str(goals)))
    if goal_seed is not None:
        params.append(("goal_seed", goal_seed))
    return params


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
    """Shared validation and request building for both clients.

    Every ``_*_call`` method returns the ``(method, path, payload)`` triple a
    public client method sends; all validation lives here so the synchronous
    and asynchronous clients cannot drift apart.
    """

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

    def _health_call(self) -> Call:
        """The server health document."""

        return "GET", "/health", None

    def _context_call(
        self, *, goals: int | None = None, goal_seed: str | None = None,
    ) -> Call:
        """The latest context bundle, read without synthesizing an answer.

        `goals` and `goal_seed` steer the server's learning-goal sample; both
        default to the server's own behavior (a random three to five goals,
        seeded from the bundle version). See `prepare_turn`.
        """

        params = _goal_params(goals, goal_seed)
        path = self._space_path("context")
        return "GET", f"{path}?{urlencode(params)}" if params else path, None

    def prepare_turn(
        self,
        message: Any,
        *,
        context_version: str | None = None,
        memory_limit: int = 5,
        source: Any = None,
        idempotency_key: str | None = None,
        goals: int | None = None,
        goal_seed: str | None = None,
    ) -> Json:
        """Build a validated turns payload from one message or a list of them.

        Messages may be authored by ``user`` or ``assistant`` (the author
        defaults to ``user``), matching the server's single write endpoint: a
        batch whose last message is not from the user is persisted without
        context/recall enrichment.

        `goals` and `goal_seed` steer the learning-goal sample in the
        returned guidance and context bundle. Both default to `None`, the
        server's historical behavior: a random three to five goals, seeded
        from the context bundle version -- which means they reshuffle on
        every version bump. Pass `goals=0` for hypotheses only, `goals=n`
        for exactly n, and a stable `goal_seed` (per conversation, per day)
        to hold the selection still across version bumps.
        """
        if (not isinstance(memory_limit, int) or isinstance(memory_limit, bool)
                or not 1 <= memory_limit <= 10):
            raise ValueError("memory_limit must be between 1 and 10")
        if context_version is not None and not str(context_version).strip():
            raise ValueError("context_version must be non-empty when provided")
        _validate_goals(goals)
        payload: Json = {
            "messages": _normalise_turn_messages(
                message, source=source, idempotency_key=idempotency_key),
            "context_version": context_version,
            "memory_limit": memory_limit,
        }
        if goals is not None:
            payload["goals"] = goals
        if goal_seed is not None:
            payload["goal_seed"] = goal_seed
        return payload

    def _turn_call(self, message: Any, **kwargs: Any) -> Call:
        """Persist a turn batch and read back context/recall metadata."""

        return "POST", self._space_path("turns"), self.prepare_turn(message, **kwargs)

    def _query_call(
        self,
        question: str | Mapping[str, Any],
        *,
        people: Sequence[str] | None = None,
        include_relationships: bool = True,
        expand_relationships: int = 0,
        include_evidence: bool = True,
        limit: int = 30,
        **extra: Any,
    ) -> Call:
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
        return "POST", self._space_path("query"), payload

    def _add_memory_call(
        self,
        content: str | Mapping[str, Any],
        *,
        kind: str = "fact",
        people: Sequence[Mapping[str, Any] | Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        **extra: Any,
    ) -> Call:
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
        return "POST", self._space_path("memories"), payload

    def _retract_call(self, memory_id: str, *, reason: str | None = None) -> Call:
        """Retract an active memory while retaining its provenance."""

        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id is required")
        payload = {"reason": reason} if reason is not None else {}
        identifier = quote(memory_id, safe="")
        return "POST", self._space_path(f"memories/{identifier}/retract"), payload

    def _supersede_call(
        self,
        memory_id: str,
        content: str,
        *,
        kind: str | None = None,
        reason: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> Call:
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
        return "POST", self._space_path(f"memories/{identifier}/supersede"), payload

    def _recall_call(
        self,
        q: str,
        *,
        about_user: bool | None = None,
        person_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> Call:
        """Cheap, LLM-free keyword search over stored memories."""

        params: list[tuple[str, str]] = [("q", q), ("limit", str(limit))]
        if about_user is not None:
            params.append(("about_user", "true" if about_user else "false"))
        for person_id in person_ids or []:
            params.append(("person_id", person_id))
        return "GET", f"{self._space_path('memories')}?{urlencode(params)}", None

    def _guidance_call(
        self,
        *,
        person_ids: Sequence[str] | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> Call:
        """Open hypotheses and open/partial learning goals, shuffled on every call.

        Unlike the small sample riding along in `context()`/`turn()`
        (deliberately KV-cache-friendly), this returns everything matching
        the focus filter -- for an agent that explicitly asks what it still
        wants to find out about the user or about given people.
        """

        params: list[tuple[str, str]] = [("limit", str(limit))]
        for person_id in person_ids or []:
            params.append(("person_id", person_id))
        if kind is not None:
            params.append(("kind", kind))
        return "GET", f"{self._space_path('guidance')}?{urlencode(params)}", None

    def _list_people_call(self, q: str = "", *, limit: int = 50) -> Call:
        """List/search known people (id, display name, aliases) -- no LLM.

        This is the discovery step before `merge_person`: it is the only
        way to see that two similar person records exist.
        """

        params = {"q": q, "limit": str(limit)}
        return "GET", f"{self._space_path('people')}?{urlencode(params)}", None

    def _person_dossier_call(self, person_id: str) -> Call:
        """Read a person projection without synthesis."""

        if not person_id:
            raise ValueError("person_id is required")
        return "GET", self._space_path(f"people/{quote(person_id, safe='')}"), None

    def _merge_person_call(self, source_person_id: str, target_person_id: str) -> Call:
        """Merge a confirmed source person into the canonical target person."""

        if not str(source_person_id).strip() or not str(target_person_id).strip():
            raise ValueError("source_person_id and target_person_id are required")
        return (
            "POST",
            self._space_path(f"people/{quote(str(source_person_id), safe='')}/merge"),
            {"target_person_id": str(target_person_id)},
        )

    def _relationship_dossier_call(self, relationship_id: str) -> Call:
        """Read a relationship projection without synthesis."""

        if not relationship_id:
            raise ValueError("relationship_id is required")
        return (
            "GET",
            self._space_path(f"relationships/{quote(relationship_id, safe='')}"),
            None,
        )


class GossipMemo(_ClientCommon):
    """Synchronous GossipMemo client.

    Parameters are deliberately explicit so an integration cannot silently
    send memories to the wrong space.  A custom ``httpx.Client`` or transport
    can be supplied for connection pooling and tests; owned clients are closed
    by :meth:`close` and custom clients are left open.

    Each method below builds its request with the matching ``_*_call`` helper
    on :class:`_ClientCommon`, which documents the request's semantics.
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

        return self._request(*self._health_call())

    def context(
        self, *, goals: int | None = None, goal_seed: str | None = None,
    ) -> Any:
        """Read the latest context bundle without synthesizing an answer."""

        return self._request(*self._context_call(goals=goals, goal_seed=goal_seed))

    def turn(self, message: Any, **kwargs: Any) -> Any:
        """Persist one turn batch and return context/recall metadata.

        Accepts `prepare_turn`'s keywords, including `goals`/`goal_seed`.
        """

        return self._request(*self._turn_call(message, **kwargs))

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

        return self._request(*self._query_call(
            question,
            people=people,
            include_relationships=include_relationships,
            expand_relationships=expand_relationships,
            include_evidence=include_evidence,
            limit=limit,
            **extra,
        ))

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

        return self._request(*self._add_memory_call(
            content,
            kind=kind,
            people=people,
            valid_from=valid_from,
            valid_to=valid_to,
            **extra,
        ))

    def retract(self, memory_id: str, *, reason: str | None = None) -> Any:
        """Retract an active memory while retaining its provenance."""

        return self._request(*self._retract_call(memory_id, reason=reason))

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

        return self._request(*self._supersede_call(
            memory_id,
            content,
            kind=kind,
            reason=reason,
            valid_from=valid_from,
            valid_to=valid_to,
        ))

    def recall(
        self,
        q: str,
        *,
        about_user: bool | None = None,
        person_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> Any:
        """Cheap, LLM-free keyword search over stored memories."""

        return self._request(*self._recall_call(
            q, about_user=about_user, person_ids=person_ids, limit=limit))

    def guidance(
        self,
        *,
        person_ids: Sequence[str] | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> Any:
        """Open hypotheses and open/partial learning goals, shuffled on every call."""

        return self._request(*self._guidance_call(
            person_ids=person_ids, kind=kind, limit=limit))

    def list_people(self, q: str = "", *, limit: int = 50) -> Any:
        """List/search known people (id, display name, aliases) -- no LLM."""

        return self._request(*self._list_people_call(q, limit=limit))

    def person_dossier(self, person_id: str) -> Any:
        """Read a person projection without synthesis."""

        return self._request(*self._person_dossier_call(person_id))

    def merge_person(self, source_person_id: str, target_person_id: str) -> Any:
        """Merge a confirmed source person into the canonical target person."""

        return self._request(*self._merge_person_call(
            source_person_id, target_person_id))

    def relationship_dossier(self, relationship_id: str) -> Any:
        """Read a relationship projection without synthesis."""

        return self._request(*self._relationship_dossier_call(relationship_id))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GossipMemo:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncGossipMemo(_ClientCommon):
    """Asynchronous counterpart to :class:`GossipMemo`.

    The methods mirror the synchronous client exactly, sharing its request
    building and validation; only awaiting differs.
    """

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
        """Return the server health document."""

        return await self._request(*self._health_call())

    async def context(
        self, *, goals: int | None = None, goal_seed: str | None = None,
    ) -> Any:
        """Read the latest context bundle without synthesizing an answer."""

        return await self._request(*self._context_call(goals=goals, goal_seed=goal_seed))

    async def turn(self, message: Any, **kwargs: Any) -> Any:
        """Persist one turn batch and return context/recall metadata.

        Accepts `prepare_turn`'s keywords, including `goals`/`goal_seed`.
        """

        return await self._request(*self._turn_call(message, **kwargs))

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
        """Query memories and relationship/person projections."""

        return await self._request(*self._query_call(
            question,
            people=people,
            include_relationships=include_relationships,
            expand_relationships=expand_relationships,
            include_evidence=include_evidence,
            limit=limit,
            **extra,
        ))

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
        """Create a manual memory in the configured space."""

        return await self._request(*self._add_memory_call(
            content,
            kind=kind,
            people=people,
            valid_from=valid_from,
            valid_to=valid_to,
            **extra,
        ))

    async def retract(self, memory_id: str, *, reason: str | None = None) -> Any:
        """Retract an active memory while retaining its provenance."""

        return await self._request(*self._retract_call(memory_id, reason=reason))

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
        """Replace a memory while preserving its correction history."""

        return await self._request(*self._supersede_call(
            memory_id,
            content,
            kind=kind,
            reason=reason,
            valid_from=valid_from,
            valid_to=valid_to,
        ))

    async def recall(
        self,
        q: str,
        *,
        about_user: bool | None = None,
        person_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> Any:
        """Cheap, LLM-free keyword search over stored memories."""

        return await self._request(*self._recall_call(
            q, about_user=about_user, person_ids=person_ids, limit=limit))

    async def guidance(
        self,
        *,
        person_ids: Sequence[str] | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> Any:
        """Open hypotheses and open/partial learning goals, shuffled on every call."""

        return await self._request(*self._guidance_call(
            person_ids=person_ids, kind=kind, limit=limit))

    async def list_people(self, q: str = "", *, limit: int = 50) -> Any:
        """List/search known people (id, display name, aliases) -- no LLM."""

        return await self._request(*self._list_people_call(q, limit=limit))

    async def person_dossier(self, person_id: str) -> Any:
        """Read a person projection without synthesis."""

        return await self._request(*self._person_dossier_call(person_id))

    async def merge_person(self, source_person_id: str, target_person_id: str) -> Any:
        """Merge a confirmed source person into the canonical target person."""

        return await self._request(*self._merge_person_call(
            source_person_id, target_person_id))

    async def relationship_dossier(self, relationship_id: str) -> Any:
        """Read a relationship projection without synthesis."""

        return await self._request(*self._relationship_dossier_call(relationship_id))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncGossipMemo:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


__all__ = [
    "AsyncGossipMemo",
    "GossipMemo",
    "GossipMemoError",
]
