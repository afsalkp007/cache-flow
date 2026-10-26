"""Unit tests for CacheFlow Pydantic schemas."""

import pytest
from cache_flow.models import CacheEntry, CacheLookupResult, CacheStats


def test_cache_entry_model():
    entry = CacheEntry(
        key_id="test1234",
        query="How to reset password?",
        embedding=[0.1] * 256,
        response="Go to settings.",
        token_count=35,
    )
    assert entry.key_id == "test1234"
    assert entry.hit_count == 0
    assert len(entry.embedding) == 256


def test_cache_lookup_result_model():
    result = CacheLookupResult(
        is_hit=True,
        similarity=0.92,
        matched_query="Prior query",
        response="Cached answer",
        tokens_saved=50,
        cost_saved_usd=0.0000075,
        latency_ms=1.2,
        source="CACHE_HIT",
    )
    assert result.is_hit is True
    assert result.tokens_saved == 50


def test_cache_stats_defaults():
    stats = CacheStats()
    assert stats.total_queries == 0
    assert stats.hit_ratio_percent == 0.0
    assert stats.speedup_factor == 1.0
