"""Tests for the `embedding.py` transport seam.

All offline: HTTP is stubbed with `httpx.MockTransport`, matching the
pattern `test_llm_transport.py` uses for the chat-completions adapter.
"""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from gossipmemo.embedding import (
    MAX_EMBEDDING_INPUT_CHARS,
    EmbeddingDimensionError,
    EmbeddingProtocolError,
    EmbeddingRequestError,
    OpenAICompatibleEmbeddingClient,
    normalize,
    resolve_embedding_dim,
)
from tests.fakes_embedding import FakeEmbeddingClient, deterministic_unit_vector


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


# --- batching / request shape -------------------------------------------


def test_embed_sends_whole_batch_in_one_request() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 2, "embedding": [3.0, 4.0]},
                ]
            },
        )

    async def run() -> list[list[float]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "", "embed-model", 2, client=http_client
            )
            return await client.embed(["a", "b", "c"])

    result = asyncio.run(run())

    assert len(captured) == 1
    assert captured[0]["model"] == "embed-model"
    assert captured[0]["input"] == ["a", "b", "c"]
    # Reordered by response `index`, and each vector is L2-normalized.
    assert result[2] == pytest.approx([0.6, 0.8])
    for vector in result:
        assert _norm(vector) == pytest.approx(1.0)


def test_embed_empty_batch_makes_no_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    async def run() -> list[list[float]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "k", "m", 4, client=http_client
            )
            return await client.embed([])

    result = asyncio.run(run())
    assert result == []
    assert calls == 0


# --- no api key -> no Authorization header --------------------------------


def test_embed_without_api_key_sends_no_authorization_header() -> None:
    seen_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "", "m", 2, client=http_client
            )
            await client.embed(["hi"])

    asyncio.run(run())
    assert "authorization" not in seen_headers[0]


def test_embed_with_api_key_sends_bearer_header() -> None:
    seen_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "secret", "m", 2, client=http_client
            )
            await client.embed(["hi"])

    asyncio.run(run())
    assert seen_headers[0]["authorization"] == "Bearer secret"


# --- request/transport failures -------------------------------------------


def test_embed_raises_on_http_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "", "m", 2, client=http_client
            )
            await client.embed(["hi"])

    with pytest.raises(EmbeddingRequestError, match="boom"):
        asyncio.run(run())


def test_embed_raises_protocol_error_on_mismatched_batch_size() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "", "m", 1, client=http_client
            )
            await client.embed(["a", "b"])

    with pytest.raises(EmbeddingProtocolError):
        asyncio.run(run())


# --- dimension probing ------------------------------------------------


def test_resolve_embedding_dim_uses_probe_when_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "qwen3-embedding-0.6b", "meta": {"n_embd": 1024}}]},
        )

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await resolve_embedding_dim(
                "http://x", "", "qwen3-embedding-0.6b", None, client=http_client
            )

    assert asyncio.run(run()) == 1024


def test_resolve_embedding_dim_falls_back_to_config_when_probe_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await resolve_embedding_dim(
                "http://x", "", "qwen3-embedding-0.6b", 1024, client=http_client
            )

    assert asyncio.run(run()) == 1024


def test_resolve_embedding_dim_falls_back_when_meta_missing() -> None:
    """`meta` is a llama.cpp extension -- absence must not crash, only degrade."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "qwen3-embedding-0.6b"}]})

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await resolve_embedding_dim(
                "http://x", "", "qwen3-embedding-0.6b", 1024, client=http_client
            )

    assert asyncio.run(run()) == 1024


def test_resolve_embedding_dim_raises_on_conflict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "qwen3-embedding-0.6b", "meta": {"n_embd": 1024}}]},
        )

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await resolve_embedding_dim(
                "http://x", "", "qwen3-embedding-0.6b", 768, client=http_client
            )

    with pytest.raises(EmbeddingDimensionError):
        asyncio.run(run())


def test_resolve_embedding_dim_raises_when_nothing_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await resolve_embedding_dim(
                "http://x", "", "qwen3-embedding-0.6b", None, client=http_client
            )

    with pytest.raises(EmbeddingDimensionError):
        asyncio.run(run())


# --- truncation -------------------------------------------------------


def test_embed_truncates_oversized_input_and_logs(caplog) -> None:
    long_text = "x" * (MAX_EMBEDDING_INPUT_CHARS + 500)
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "", "m", 2, client=http_client
            )
            with caplog.at_level("WARNING"):
                await client.embed([long_text])

    asyncio.run(run())
    assert len(captured[0]["input"][0]) == MAX_EMBEDDING_INPUT_CHARS
    assert "embedding_input_truncated" in caplog.text


# --- normalization ------------------------------------------------------


def test_normalize_is_idempotent() -> None:
    vector = [3.0, 4.0, 0.0]
    once = normalize(vector)
    twice = normalize(once)
    assert once == pytest.approx(twice)
    assert _norm(once) == pytest.approx(1.0)


def test_normalize_handles_zero_vector_safely() -> None:
    assert normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


# --- instruction prefix (ingestion never sets it; interface allows it) ----


def test_deterministic_unit_vector_is_reproducible_and_unit_norm() -> None:
    first = deterministic_unit_vector("hello", 16)
    second = deterministic_unit_vector("hello", 16)
    assert first == second
    assert len(first) == 16
    assert _norm(first) == pytest.approx(1.0)


def test_fake_embedding_client_applies_instruction_prefix_and_changes_vector() -> None:
    fake = FakeEmbeddingClient(dim=8)

    async def run() -> tuple[list[list[float]], list[list[float]]]:
        plain = await fake.embed(["hello"])
        prefixed = await fake.embed(["hello"], instruction="Find related notes")
        return plain, prefixed

    plain, prefixed = asyncio.run(run())

    assert plain != prefixed
    assert fake.calls[0] == (("hello",), None)
    assert fake.calls[1] == (("hello",), "Find related notes")


def test_embedding_client_embed_accepts_optional_instruction_kw() -> None:
    """Interface check: `embed` takes an optional `instruction` kw, defaulting to None."""

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleEmbeddingClient(
                "http://x", "", "m", 2, client=http_client
            )
            await client.embed(["find my keys"], instruction="Retrieve relevant memory")

    asyncio.run(run())
    assert captured[0]["input"] == ["Instruct: Retrieve relevant memory\nQuery: find my keys"]


def test_endpoint_treats_base_url_as_the_api_root() -> None:
    """`base_url` already carries `/v1`, exactly as `llm.py` treats `llm_base_url`.

    That convention is what makes inheriting `llm_base_url` for embedding
    correct; appending `/v1/embeddings` here would double the prefix.
    """

    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            for base in ("http://nas:8002/v1", "http://nas:8002/v1/"):
                client = OpenAICompatibleEmbeddingClient(
                    base, "", "embed-model", 2, client=http_client
                )
                await client.embed(["a"])

    asyncio.run(run())
    assert urls == ["http://nas:8002/v1/embeddings"] * 2


def test_dimension_probe_hits_the_models_path_under_the_api_root() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(
            200, json={"data": [{"id": "embed-model", "meta": {"n_embd": 1024}}]}
        )

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await resolve_embedding_dim(
                "http://nas:8002/v1", "", "embed-model", None, client=http_client
            )

    assert asyncio.run(run()) == 1024
    assert urls == ["http://nas:8002/v1/models"]
