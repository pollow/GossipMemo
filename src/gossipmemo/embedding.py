"""Embedding transport: one batch embeddings request, and nothing above it.

Mirrors `transport.py`'s discipline for the chat-completions seam: this is a
leaf module. It owns HTTP details for an OpenAI-compatible `/embeddings`
endpoint (and the llama.cpp `/models` dimension probe) and nothing about
which store rows get embedded, when a backfill runs, or how vectors are
searched -- that is a later slice's job. Nothing here imports a reasoner, a
prompt, or a store, and this client is never routed through `ProviderGate`:
that gate protects a single shared remote chat provider with a global
backoff that deliberately blocks every caller, and embedding here talks to
an independent (usually local) provider whose availability has nothing to
do with the chat provider's.

Ingestion-side discipline: `embed()` encodes text for storage only. It never
adds an instruction prefix. Query-side asymmetric prefixing is a later
slice's concern, but the optional `instruction` parameter on `embed()` lets
that slice ask for one without this module knowing who calls it or why.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

# Default when no `Settings` is available to supply
# `embedding_query_timeout_seconds` (e.g. a test injects a client directly).
# Deliberately short and independent of `llm_timeout_seconds`: a query-side
# embedding call sits on a foreground request path (a turn or a `/query`
# call), and must never hold that request hostage to a slow or wedged
# embedding provider the way a 120s LLM timeout would.
DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS = 2.0

# Conservative character cap applied before a text is sent for embedding.
# The deployed provider is llama.cpp with n_ctx=4096 tokens; this is a
# comfortably-under-budget character bound (well short of the ~4x
# chars-per-token ratio one would assume for English, and lower still for
# dense CJK text) rather than a token-accurate count -- accuracy here is not
# worth pulling in a tokenizer for a leaf transport module.
MAX_EMBEDDING_INPUT_CHARS = 6000


class EmbeddingError(RuntimeError):
    """Base class for embedding configuration, transport, and protocol failures."""


class EmbeddingRequestError(EmbeddingError):
    """Raised when the embeddings HTTP request fails."""


class EmbeddingProtocolError(EmbeddingError):
    """Raised when a provider response is not a valid embeddings response."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when the embedding dimension cannot be determined, or conflicts."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """What a caller needs to turn text into vectors, and nothing it doesn't."""

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> list[list[float]]: ...


def _truncate(text: str) -> str:
    if len(text) <= MAX_EMBEDDING_INPUT_CHARS:
        return text
    logger.warning(
        "embedding_input_truncated",
        extra={"original_chars": len(text), "limit_chars": MAX_EMBEDDING_INPUT_CHARS},
    )
    return text[:MAX_EMBEDDING_INPUT_CHARS]


def _apply_instruction(text: str, instruction: str | None) -> str:
    if instruction is None:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize a vector, idempotently and safely for an all-zero input.

    We do not trust a provider's claim to already return normalized
    vectors, so this is applied unconditionally before a vector is stored.
    Applying it twice is a no-op (within floating-point error).
    """

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [component / norm for component in vector]


async def probe_dimension(
    client: httpx.AsyncClient, base_url: str, headers: Mapping[str, str], model: str
) -> int | None:
    """Probe `GET {base_url}/models` for the configured model's `n_embd`.

    `meta.n_embd` is a llama.cpp server extension, not something the OpenAI
    protocol guarantees, so every step here is defensive: a missing field,
    an unexpected shape, or a transport failure all mean "probe failed"
    (`None`), never an exception. The caller decides what a failed probe
    means for startup.
    """

    try:
        response = await client.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("id") != model:
            continue
        meta = entry.get("meta")
        if not isinstance(meta, Mapping):
            return None
        n_embd = meta.get("n_embd")
        if isinstance(n_embd, int) and n_embd > 0:
            return n_embd
        return None
    return None


class OpenAICompatibleEmbeddingClient:
    """Client for servers exposing an OpenAI-compatible `/embeddings`.

    One HTTP request per `embed()` call carries the whole batch, matching
    the deployed llama.cpp server's support for multiple `input` strings.
    `api_key` may be empty (the deployed NAS server needs none); when empty,
    no `Authorization` header is sent at all rather than sending an empty
    bearer token.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_base = base_url.strip().rstrip("/")
        if not normalized_base:
            raise ValueError("embedding base_url must not be empty")
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        if dim <= 0:
            raise ValueError("embedding dim must be greater than zero")
        if timeout <= 0:
            raise ValueError("embedding timeout must be greater than zero")

        self.base_url = normalized_base
        self.api_key = api_key.strip()
        self._model = model.strip()
        self._dim = dim
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return self.base_url + "/embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def embed(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> list[list[float]]:
        """Embed a batch of texts in a single request, returning L2-normalized vectors.

        Storage-side callers must not pass `instruction` -- that prefix
        exists only for the query side of a later slice.
        """

        if not texts:
            return []
        inputs = [_apply_instruction(_truncate(text), instruction) for text in texts]
        client = await self._get_client()
        try:
            response = await client.post(
                self.endpoint,
                headers=self._headers(),
                json={"model": self._model, "input": inputs},
            )
        except httpx.TransportError as error:
            raise EmbeddingRequestError(f"embedding request failed: {error}") from error
        if response.is_error:
            detail = _response_detail(response)
            raise EmbeddingRequestError(
                f"embedding request failed with HTTP {response.status_code}: {detail}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise EmbeddingProtocolError(
                "embedding response was not valid JSON"
            ) from error
        vectors = _parse_embeddings_response(payload, expected=len(inputs))
        return [normalize(vector) for vector in vectors]

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await client.aclose()

    async def __aenter__(self) -> OpenAICompatibleEmbeddingClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _parse_embeddings_response(payload: Any, *, expected: int) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise EmbeddingProtocolError("embedding response was not a JSON object")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise EmbeddingProtocolError(
            "embedding response 'data' did not match the requested batch size"
        )
    indexed: list[tuple[int, list[float]]] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            raise EmbeddingProtocolError("embedding response entry was not a JSON object")
        embedding = entry.get("embedding")
        index = entry.get("index")
        if not isinstance(embedding, list) or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise EmbeddingProtocolError("embedding response entry had no valid 'embedding'")
        if not isinstance(index, int):
            raise EmbeddingProtocolError("embedding response entry had no valid 'index'")
        indexed.append((index, [float(value) for value in embedding]))
    indexed.sort(key=lambda pair: pair[0])
    return [vector for _, vector in indexed]


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        detail = response.text
    else:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or str(error)
            else:
                detail = payload.get("message") or str(payload)
        else:
            detail = str(payload)
    return str(detail).strip()[:1000] or "no response body"


async def embed_query_vector(
    client: EmbeddingClient | None,
    text: str,
    *,
    instruction: str,
    timeout: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> list[float] | None:
    """Best-effort single-text query embedding, for the hybrid-retrieval slice.

    Every failure mode -- no client configured, a request/protocol error, or
    exceeding `timeout` -- degrades to `None` rather than raising: it is
    always this call's job to fall back to plain FTS, never to fail the
    caller's request. `instruction` is the asymmetric query-side prefix
    (see `_apply_instruction`); storage-side embedding must never go
    through this function.
    """

    if client is None:
        return None
    try:
        vectors = await asyncio.wait_for(
            client.embed([text], instruction=instruction), timeout=timeout,
        )
    except Exception:
        logger.warning("embedding_query_failed", exc_info=True)
        return None
    if not vectors:
        return None
    return vectors[0]


async def resolve_embedding_dim(
    base_url: str, api_key: str, model: str, configured_dim: int | None,
    *, client: httpx.AsyncClient | None = None, timeout: float = 120.0,
) -> int:
    """Resolve the embedding dimension by probing, falling back to config.

    Probe `GET {base_url}/models` first. If the probe succeeds and
    `configured_dim` is also set, they must agree -- a mismatch is a
    configuration error, since a silently-wrong dimension would corrupt
    every stored vector. If the probe fails, fall back to `configured_dim`.
    If both are unavailable, refuse: there is no dimension to build a
    client around.
    """

    owns_client = client is None
    probe_client = client or httpx.AsyncClient(timeout=timeout)
    headers = {"Accept": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        probed = await probe_dimension(probe_client, base_url.rstrip("/"), headers, model)
    finally:
        if owns_client:
            await probe_client.aclose()

    if probed is not None and configured_dim is not None and probed != configured_dim:
        raise EmbeddingDimensionError(
            f"probed embedding dim {probed} conflicts with configured "
            f"GOSSIPMEMO_EMBEDDING_DIM={configured_dim}"
        )
    if probed is not None:
        return probed
    if configured_dim is not None:
        return configured_dim
    raise EmbeddingDimensionError(
        "could not determine embedding dimension: dimension probe failed and "
        "GOSSIPMEMO_EMBEDDING_DIM is not set"
    )


async def create_embedding_client(
    settings: Settings, *, client: httpx.AsyncClient | None = None
) -> OpenAICompatibleEmbeddingClient | None:
    """Build the embedding client from validated process settings.

    Returns `None` when `settings.embedding_enabled` is false -- embedding
    is a legitimate off state, not a startup failure.
    """

    if not settings.embedding_enabled:
        return None
    dim = await resolve_embedding_dim(
        settings.embedding_base_url,
        settings.embedding_api_key,
        settings.embedding_model,
        settings.embedding_dim,
        client=client,
        timeout=settings.llm_timeout_seconds,
    )
    return OpenAICompatibleEmbeddingClient(
        settings.embedding_base_url,
        settings.embedding_api_key,
        settings.embedding_model,
        dim,
        timeout=settings.llm_timeout_seconds,
        client=client,
    )


__all__ = [
    "DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS",
    "EmbeddingClient",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProtocolError",
    "EmbeddingRequestError",
    "MAX_EMBEDDING_INPUT_CHARS",
    "OpenAICompatibleEmbeddingClient",
    "create_embedding_client",
    "embed_query_vector",
    "normalize",
    "probe_dimension",
    "resolve_embedding_dim",
]
