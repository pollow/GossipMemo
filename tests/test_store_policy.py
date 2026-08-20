"""Direct unit tests for the pure store policy.

These rules used to live inside `SqliteWorldStore` and were only reachable
through SQL fixtures. They are exercised here on their own terms.
"""

from __future__ import annotations

from gossipmemo.models import GuidanceItem
from gossipmemo.store.policy import (
    fts_query,
    is_profile_stale,
    rank_guidance,
    sample_learning_goals,
    similar_memory_content,
)


def _goal(identifier: str, content: str = "goal") -> GuidanceItem:
    return GuidanceItem(
        id=identifier, kind="learning_goal", content=content, owner_kind="user", status="open"
    )


def _hypothesis(identifier: str, content: str) -> GuidanceItem:
    return GuidanceItem(
        id=identifier, kind="hypothesis", content=content, owner_kind="user", status="open"
    )


# --- similar_memory_content ------------------------------------------------


def test_similar_memory_content_matches_identical_text():
    assert similar_memory_content("Alice drinks oat milk", "Alice drinks oat milk")


def test_similar_memory_content_ignores_whitespace_punctuation_and_nfkc_width():
    assert similar_memory_content("Alice drinks oat milk", "  alice   drinks, oat milk!  ")
    assert similar_memory_content("他喜欢喝咖啡", "他喜欢喝咖啡")
    # Fullwidth latin normalizes to ASCII under NFKC.
    assert similar_memory_content("Alice drinks oat milk", "Ａlice drinks oat milk")


def test_similar_memory_content_keeps_differing_numbers_apart():
    assert not similar_memory_content(
        "Alice has worked at Acme for 3 years", "Alice has worked at Acme for 4 years"
    )


def test_similar_memory_content_keeps_english_negation_apart():
    assert not similar_memory_content(
        "Alice likes running in the morning", "Alice does not like running in the morning"
    )


def test_similar_memory_content_keeps_chinese_negation_apart():
    assert not similar_memory_content("他喜欢喝咖啡因为提神", "他不喜欢喝咖啡因为提神")


def test_similar_memory_content_threshold_is_conservative():
    base = "Alice works as a senior backend engineer at Acme Corporation"
    # ratio ~0.99: one trailing character dropped.
    assert similar_memory_content(base, base[:-1])
    # ratio ~0.93: a changed final word is a different claim.
    assert not similar_memory_content(
        base, "Alice works as a senior backend engineer at Acme Company"
    )


# --- fts_query -------------------------------------------------------------


def test_fts_query_returns_none_for_the_dossier_and_reason_questions():
    assert fts_query("dossier") is None
    assert fts_query("  Dossier  ") is None
    assert fts_query("reason") is None


def test_fts_query_drops_short_tokens_and_excluded_terms():
    assert fts_query("Who is Alice at the cafe?", ["Alice"]) == '"who" OR "the" OR "cafe"'


def test_fts_query_returns_none_when_no_term_survives():
    assert fts_query("a an is") is None
    assert fts_query("") is None


def test_fts_query_expands_cjk_runs_into_trigrams():
    assert fts_query("他喜欢喝咖啡") == '"他喜欢" OR "喜欢喝" OR "欢喝咖" OR "喝咖啡"'


def test_fts_query_keeps_short_cjk_runs_whole_and_deduplicates():
    assert fts_query("咖啡 咖啡 咖啡馆") == '"咖啡馆"'


# --- is_profile_stale ------------------------------------------------------


def test_profile_without_any_source_memory_is_never_stale():
    assert not is_profile_stale(None, None)
    assert not is_profile_stale("2026-01-01T00:00:00+00:00", None)


def test_profile_is_stale_when_the_watermark_moved():
    assert is_profile_stale(None, "2026-01-01T00:00:00+00:00")
    assert is_profile_stale("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00")
    assert not is_profile_stale("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")


# --- rank_guidance ---------------------------------------------------------


def test_exact_substring_outranks_bigram_overlap():
    exact = _hypothesis("h_exact", "She may be planning a trip to Kyoto")
    overlap = _hypothesis("h_overlap", "trip trip trip tri pt ri pto")
    updated = {"h_exact": "2026-01-01T00:00:00+00:00", "h_overlap": "2026-09-01T00:00:00+00:00"}
    ranked = rank_guidance([overlap, exact], updated, "trip to kyoto")
    assert [item.id for item in ranked] == ["h_exact", "h_overlap"]


def test_recency_breaks_ties_between_equally_relevant_items():
    older = _hypothesis("h_old", "She may be planning a trip")
    newer = _hypothesis("h_new", "She may be planning a trip")
    updated = {"h_old": "2026-01-01T00:00:00+00:00", "h_new": "2026-09-01T00:00:00+00:00"}
    ranked = rank_guidance([older, newer], updated, "unrelated wording")
    assert [item.id for item in ranked] == ["h_new", "h_old"]


def test_ranking_without_a_query_falls_back_to_recency_then_id():
    items = [_hypothesis("h_a", "one"), _hypothesis("h_b", "two")]
    updated = {"h_a": "2026-01-01T00:00:00+00:00", "h_b": "2026-01-01T00:00:00+00:00"}
    assert [item.id for item in rank_guidance(items, updated, "")] == ["h_b", "h_a"]


# --- sample_learning_goals -------------------------------------------------


def _pool(size: int) -> list[GuidanceItem]:
    return [_goal(f"g_{index:02d}") for index in range(size)]


def test_sample_is_bounded_by_the_min_max_window():
    sample = sample_learning_goals(_pool(12), "v1", 3, 5)
    assert 3 <= len(sample) <= 5
    assert len({item.id for item in sample}) == len(sample)


def test_sample_never_exceeds_the_available_pool():
    assert len(sample_learning_goals(_pool(2), "v1", 3, 5)) == 2
    assert sample_learning_goals([], "v1", 3, 5) == []


def test_same_version_and_pool_yield_the_same_sample():
    pool = _pool(12)
    first = sample_learning_goals(pool, "version-a", 3, 5)
    second = sample_learning_goals(list(pool), "version-a", 3, 5)
    assert [item.id for item in first] == [item.id for item in second]


def test_changing_the_version_or_the_pool_rotates_the_sample():
    pool = _pool(12)
    base = [item.id for item in sample_learning_goals(pool, "version-a", 3, 5)]
    assert [item.id for item in sample_learning_goals(pool, "version-b", 3, 5)] != base
    grown = pool + [_goal("g_99")]
    assert [item.id for item in sample_learning_goals(grown, "version-a", 3, 5)] != base
