"""Unit tests for LLMGateway proxy routing and ROI calculations."""

import pytest
from cache_flow.cache import SemanticCache
from cache_flow.gateway import LLMGateway, MockProvider


@pytest.fixture
def gateway():
    cache = SemanticCache(similarity_threshold=0.70)
    return LLMGateway(cache=cache, provider=MockProvider(), cost_per_1k_tokens=0.00015)


def test_gateway_miss_then_hit_lifecycle(gateway):
    # Call 1: Miss
    r1 = gateway.query("How do I reset my account password?")
    assert r1.is_hit is False
    assert r1.source == "LLM_INFERENCE"
    assert r1.tokens_saved == 0

    # Call 2: Hit on paraphrased question
    r2 = gateway.query("What are the steps to reset password?")
    assert r2.is_hit is True
    assert r2.source == "CACHE_HIT"
    assert r2.tokens_saved > 0
    assert r2.cost_saved_usd > 0
    assert r2.latency_ms < 50.0  # In-memory lookup should be sub-50ms


def test_gateway_stats_aggregation(gateway):
    gateway.query("How to reset password?")
    gateway.query("Steps to reset password?")

    stats = gateway.get_stats()
    assert stats.total_queries == 2
    assert stats.cache_hits == 1
    assert stats.cache_misses == 1
    assert stats.hit_ratio_percent == 50.0
    assert stats.total_tokens_saved > 0
    assert stats.speedup_factor > 1.0


def test_gateway_reset_stats(gateway):
    gateway.query("How to reset password?")
    gateway.reset_stats()
    stats = gateway.get_stats()
    assert stats.total_queries == 0
    assert stats.cache_hits == 0
