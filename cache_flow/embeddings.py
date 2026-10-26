"""Semantic embedding engine and cosine similarity vector operations."""

import re
import hashlib
from collections import Counter
from typing import List, Set
import numpy as np


class SemanticEmbedder:
    """High-speed, deterministic semantic feature vectorizer with n-gram subword hashing."""

    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "over", "after",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "can", "could", "should", "would",
        "my", "your", "his", "her", "its", "our", "their", "this", "that",
        "how", "what", "where", "when", "why", "who", "which", "please",
        "tell", "me", "i", "you", "we", "they", "want", "need", "like",
    }

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _extract_features(self, text: str) -> Counter:
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        words = [w for w in clean.split() if w not in self.STOPWORDS and len(w) > 1]

        if not words:
            # If all words were stopwords, fallback to full words
            words = [w for w in clean.split() if len(w) > 0]

        features: List[str] = []
        for w in words:
            features.append(w)
            # Character n-grams (3 and 4) capture root morphology and suffixes
            for n in (3, 4):
                if len(w) >= n:
                    for i in range(len(w) - n + 1):
                        features.append(w[i : i + n])

        return Counter(features)

    def embed(self, text: str) -> List[float]:
        """Convert query string into an L2-normalized dense vector."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        counts = self._extract_features(text)
        vector = np.zeros(self.dimension, dtype=np.float32)

        for term, count in counts.items():
            h = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vector[idx] += sign * count

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two unit vectors (dot product)."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, min(1.0, float(dot)))
