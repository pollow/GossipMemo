"""Pure domain policy for the world store.

Nothing here touches SQLite. These are the subtle product rules -- what
counts as a duplicate Memory, how free text becomes an FTS query, when a
projection is stale, and how guidance is ranked and sampled -- kept apart
from the persistence layer so they can be read and tested on their own.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
import uuid
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any

from ..models import GuidanceItem, utc_now

# Reciprocal Rank Fusion constants for hybrid (FTS + vector) recall. `RRF_K`
# is the standard RRF damping constant; `RRF_CANDIDATE_K` is how many
# top-ranked candidates each retrieval path contributes before fusion --
# fixed regardless of a call site's own `limit`, per the design brief.
RRF_K = 60
RRF_CANDIDATE_K = 20


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def now_iso() -> str:
    return utc_now().isoformat()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def similar_memory_content(left: str, right: str) -> bool:
    """Conservatively match near-identical Memory wording."""

    def canonical(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
        return " ".join(value.split())

    left, right = canonical(left), canonical(right)
    if left == right:
        return True
    # Preserve polarity and numeric/date changes even when wording is close.
    if re.findall(r"\d+(?:\.\d+)?", left) != re.findall(r"\d+(?:\.\d+)?", right):
        return False

    def has_negation(value: str) -> bool:
        english = re.search(r"\b(?:not|never|no)\b", value) is not None
        return english or any(token in value for token in ("不", "没", "未"))

    if has_negation(left) != has_negation(right):
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.97


def fts_query(question: str, excluded: Iterable[str] = ()) -> str | None:
    """Build a conservative FTS5 OR query from natural-language input."""

    if question.casefold().strip() in {"dossier", "reason"}:
        return None
    excluded_terms = {normalized(value) for value in excluded}
    terms: list[str] = []
    for token in re.findall(r"[^\W_]+", question.casefold(), flags=re.UNICODE):
        if len(token) < 3 or normalized(token) in excluded_terms:
            continue
        if any("\u4e00" <= char <= "\u9fff" for char in token) and len(token) > 3:
            terms.extend(token[index: index + 3] for index in range(len(token) - 2))
        else:
            terms.append(token)
    unique = list(dict.fromkeys(terms))[:16]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique) or None


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]], *, k: int = RRF_K,
) -> list[str]:
    """Fuse several best-match-first id rankings into one, by RRF.

    `score(id) = sum(1 / (k + rank))` over every ranking the id appears in,
    `rank` 0-based within that ranking. Deliberately rank-only: bm25 and
    cosine similarity sit on incomparable scales, so combining them by raw
    score would need a data-dependent calibration this design avoids
    entirely. An id absent from a ranking simply contributes nothing from
    it. Ties (most commonly an id that appears in only one ranking, or the
    trivial case of a single non-empty ranking) break by first appearance
    across the input rankings, so the result is deterministic.
    """

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            if item_id not in first_seen:
                first_seen[item_id] = counter
                counter += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))


def is_profile_stale(profile_source_updated_at: str | None, watermark: str | None) -> bool:
    """Is a projection behind the Memories it was induced from?

    A `None` watermark means there is no source Memory at all, so there is
    nothing to be stale against -- an un-induced projection over an empty
    source set stays fresh.
    """

    return watermark is not None and profile_source_updated_at != watermark


def rank_guidance(
    items: Iterable[GuidanceItem], updated_at: dict[str, str], query: str
) -> list[GuidanceItem]:
    """Order guidance by lexical relevance, then recency, then id.

    Lexical relevance is only a tie-breaker over recency. Character bigrams
    keep short/non-space-separated text useful (including CJK), while an
    exact substring hit outweighs any amount of bigram overlap.
    """

    folded = query.casefold()
    grams = {folded[i:i + 2] for i in range(max(0, len(folded) - 1))}

    def rank(item: GuidanceItem) -> tuple[int, str, str]:
        text = item.content.casefold()
        relevance = (1000 if folded and folded in text else 0) + \
            sum(text.count(g) for g in grams)
        return (relevance, updated_at[item.id], item.id)

    return sorted(items, key=rank, reverse=True)


def sample_learning_goals(
    goals: list[GuidanceItem], version: str, minimum: int, maximum: int
) -> list[GuidanceItem]:
    """Draw a deterministic sample of learning goals.

    The sample is a function of the context bundle `version` and the
    candidate pool: a local RNG is seeded from a hash of the two, so
    identical version + identical pool always produce the identical
    selection. This keeps the sample stable across repeated reads
    (KV-cache-friendly for the agent-side prompt prefix) while still
    rotating whenever the durable context or the pool actually changes.
    """

    pool_ids = sorted(item.id for item in goals)
    seed = int(hashlib.sha256(
        f"{version}|{','.join(pool_ids)}".encode()).hexdigest()[:16], 16)
    goal_rng = random.Random(seed)
    sample_size = min(len(goals), goal_rng.randint(minimum, maximum))
    return goal_rng.sample(goals, sample_size)
