"""Unit tests for semantic vectorizer and cosine similarity operations."""

import math
import pytest
from cache_flow.embeddings import SemanticEmbedder


@pytest.fixture
def embedder():
    return SemanticEmbedder(dimension=256)


def test_embed_dimension_and_normalization(embedder):
    v = embedder.embed("How do I update my billing credit card?")
    assert len(v) == 256
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-4


def test_cosine_similarity_identity(embedder):
    v1 = embedder.embed("Cancel my subscription")
    sim = SemanticEmbedder.cosine_similarity(v1, v1)
    assert abs(sim - 1.0) < 1e-4


def test_paraphrase_similarity_exceeds_unrelated(embedder):
    v_base = embedder.embed("How do I reset my account password?")
    v_para = embedder.embed("What are the steps to reset password?")
    v_diff = embedder.embed("What is the speed of light in vacuum?")

    sim_para = SemanticEmbedder.cosine_similarity(v_base, v_para)
    sim_diff = SemanticEmbedder.cosine_similarity(v_base, v_diff)

    assert sim_para > 0.65
    assert sim_diff < 0.20
    assert sim_para > sim_diff * 3
