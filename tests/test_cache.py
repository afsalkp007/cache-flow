"""Unit tests for in-memory SemanticCache vector store and LRU eviction."""

import pytest
from cache_flow.cache import SemanticCache


@pytest.fixture
def cache():
    return SemanticCache(similarity_threshold=0.70, max_entries=3)


def test_empty_cache_lookup(cache):
    entry, sim = cache.lookup("Any question")
    assert entry is None
    assert sim == 0.0


def test_exact_and_semantic_hit(cache):
    cache.insert(
        query="How do I reset my account password?",
        response="Navigate to Settings -> Security to reset password.",
        token_count=40,
    )

    # Exact match
    entry_exact, sim_exact = cache.lookup("How do I reset my account password?")
    assert entry_exact is not None
    assert sim_exact >= 0.99
    assert entry_exact.hit_count == 1

    # Paraphrased match
    entry_para, sim_para = cache.lookup("What are the steps to reset password?")
    assert entry_para is not None
    assert sim_para >= 0.70
    assert entry_para.hit_count == 2


def test_below_threshold_miss(cache):
    cache.insert(
        query="How do I reset my account password?",
        response="Go to Settings.",
        token_count=10,
    )

    entry, sim = cache.lookup("Where is the nearest coffee shop?")
    assert entry is None
    assert sim < 0.70


def test_lru_eviction_when_capacity_exceeded(cache):
    cache.insert("Q1", "R1", 10)
    cache.insert("Q2", "R2", 10)
    cache.insert("Q3", "R3", 10)
    assert cache.size() == 3

    # Access Q1 to make it recently used
    cache.lookup("Q1")

    # Insert Q4 -> Should evict Q2 (since Q1 was accessed recently)
    cache.insert("Q4", "R4", 10)
    assert cache.size() == 3

    queries = [e.query for e in cache.get_entries()]
    assert "Q1" in queries
    assert "Q4" in queries
    assert "Q2" not in queries


def test_cache_clear(cache):
    cache.insert("Q1", "R1", 10)
    cache.insert("Q2", "R2", 10)
    cache.clear()
    assert cache.size() == 0
    assert len(cache.get_entries()) == 0
