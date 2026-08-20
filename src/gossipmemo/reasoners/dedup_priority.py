"""Priority ordering for write-time dedup comparison sets.

`owner.py` (hypothesis comparisons), `learning_goals.py` (open goals), and
`coverage.py` (a root's active entries) each already hand their reasoner a
*complete* comparison set to judge a new proposal against -- and, in
owner.py's case, already trims that set lossily (full content -> ID-only
skeleton) when it does not fit the request budget. This module does not add
a new trimming mechanism: it only decides *which order* the existing,
complete set is considered in, so that the items most similar to the
evidence currently being reasoned about are the ones a caller's own budget
logic keeps in full text, or renders first.

Nothing is ever dropped here -- `similarity_priority_order` is a stable
resort of `items`, never a filter. An item whose embedding hasn't landed
yet, or that a search error hides, simply sorts to the back, still part of
the comparison set. Every degradation path (no client configured, no query
text, an embedding failure/timeout, or `store.search_vectors` raising)
falls back to `items` in their original order -- byte-for-byte the
pre-slice-5 behavior -- logged once and never raised to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TypeVar

from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient, embed_query_vector
from ..store import WorldStore

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT")

# A generous, fixed sweep rather than `len(items)`: the store's
# `search_vectors` ranks across the whole space for this owner_kind, not
# scoped to the specific owner (person/relationship/root) the caller's
# `items` belong to, so the candidates we actually care about can rank
# anywhere in a space-wide ordering. A wide k makes it likely every one of
# them is covered; any that still fall outside it simply keep their
# original relative order (never dropped -- see module docstring).
_SEARCH_K = 200


async def similarity_priority_order(
    store: WorldStore,
    space_id: str,
    owner_kind: str,
    items: Sequence[ItemT],
    item_id: Callable[[ItemT], str],
    query_text: str,
    *,
    embedding_client_getter: Callable[[], EmbeddingClient | None],
    instruction: str,
    timeout: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> list[ItemT]:
    """Stable-resort `items` by similarity to `query_text`; never drops any."""

    if len(items) <= 1 or not query_text.strip():
        return list(items)
    client = embedding_client_getter()
    if client is None:
        return list(items)
    vector = await embed_query_vector(client, query_text, instruction=instruction, timeout=timeout)
    if vector is None:
        return list(items)
    try:
        ranked = store.search_vectors(space_id, owner_kind, vector, _SEARCH_K)
    except Exception:
        logger.warning(
            "dedup_priority_search_failed", exc_info=True, extra={"owner_kind": owner_kind},
        )
        return list(items)
    rank = {owner_id: position for position, (owner_id, _score) in enumerate(ranked)}
    worst = len(ranked)
    return sorted(items, key=lambda item: rank.get(item_id(item), worst))


__all__ = ["similarity_priority_order"]
