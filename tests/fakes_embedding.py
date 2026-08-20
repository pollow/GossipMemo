"""A deterministic fake `EmbeddingClient` for offline tests.

Hashes each text (plus its optional instruction prefix) into a reproducible
unit vector -- no network, no randomness across runs. This is not meant to
carry semantic meaning: it cannot catch "the query prefix landed on the
wrong side" bugs, only structural ones (batching, dimension, normalization
plumbing). A small, separately-gated real-provider test suite is a later
concern per the design brief.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from gossipmemo.embedding import normalize


def deterministic_unit_vector(text: str, dim: int) -> list[float]:
    """Hash-derived, reproducible unit vector for a text.

    Normalized through the production `normalize`, so a fake vector is
    always a vector the storage layer would consider valid.
    """

    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        for offset in range(0, len(digest) - 1, 2):
            if len(values) >= dim:
                break
            raw = int.from_bytes(digest[offset: offset + 2], "big")
            values.append((raw / 65535.0) * 2.0 - 1.0)
        counter += 1
    return normalize(values)


class FakeEmbeddingClient:
    """In-memory `EmbeddingClient` implementation for tests."""

    def __init__(self, model: str = "fake-embedding", dim: int = 8) -> None:
        self._model = model
        self._dim = dim
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> list[list[float]]:
        self.calls.append((tuple(texts), instruction))
        keyed = [
            text if instruction is None else f"Instruct: {instruction}\nQuery: {text}"
            for text in texts
        ]
        return [deterministic_unit_vector(text, self._dim) for text in keyed]


__all__ = ["FakeEmbeddingClient", "deterministic_unit_vector"]
