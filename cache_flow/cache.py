"""In-memory semantic vector cache with cosine similarity search and LRU eviction."""

import hashlib
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from cache_flow.models import CacheEntry
from cache_flow.embeddings import SemanticEmbedder


class SemanticCache:
    """Vector similarity cache for LLM queries with LRU eviction policy."""

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        max_entries: int = 500,
        embedder: Optional[SemanticEmbedder] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.embedder = embedder or SemanticEmbedder()
        self.entries: Dict[str, CacheEntry] = {}
        self.lru_order: List[str] = []

    def _generate_key_id(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]

    def lookup(self, query: str) -> Tuple[Optional[CacheEntry], float]:
        """
        Search the semantic cache for a query with cosine similarity above threshold.
        
        Returns:
            (matched_entry, similarity_score) if hit, else (None, best_similarity_score).
        """
        if not self.entries or not query.strip():
            return None, 0.0

        query_vec = self.embedder.embed(query)
        best_entry: Optional[CacheEntry] = None
        best_similarity = 0.0

        for key_id, entry in self.entries.items():
            sim = SemanticEmbedder.cosine_similarity(query_vec, entry.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_entry and best_similarity >= self.similarity_threshold:
            # Cache Hit: update telemetry and LRU
            best_entry.hit_count += 1
            best_entry.last_accessed_at = datetime.utcnow().isoformat()

            # Refresh LRU position
            if best_entry.key_id in self.lru_order:
                self.lru_order.remove(best_entry.key_id)
            self.lru_order.append(best_entry.key_id)

            return best_entry, round(best_similarity, 3)

        return None, round(best_similarity, 3)

    def insert(
        self,
        query: str,
        response: str,
        token_count: int,
        model: str = "gemini-2.5-flash",
    ) -> CacheEntry:
        """Insert a new LLM generation into the semantic cache."""
        key_id = self._generate_key_id(query)

        # LRU Eviction if capacity exceeded
        if len(self.entries) >= self.max_entries and key_id not in self.entries:
            evict_key = self.lru_order.pop(0)
            self.entries.pop(evict_key, None)

        embedding = self.embedder.embed(query)
        entry = CacheEntry(
            key_id=key_id,
            query=query.strip(),
            embedding=embedding,
            response=response.strip(),
            token_count=token_count,
            model=model,
            hit_count=0,
        )

        self.entries[key_id] = entry
        if key_id in self.lru_order:
            self.lru_order.remove(key_id)
        self.lru_order.append(key_id)

        return entry

    def clear(self) -> None:
        """Flush the cache and reset indices."""
        self.entries.clear()
        self.lru_order.clear()

    def get_entries(self) -> List[CacheEntry]:
        """Return list of all cached entries sorted by recency."""
        return [self.entries[k] for k in reversed(self.lru_order) if k in self.entries]

    def size(self) -> int:
        """Number of active entries in cache."""
        return len(self.entries)
