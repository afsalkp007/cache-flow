"""Pydantic schemas for semantic cache entries, lookups, and real-time ROI metrics."""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """An individual cached item stored in the semantic vector index."""
    key_id: str = Field(..., description="Unique entry hash ID")
    query: str = Field(..., description="Original user prompt or question")
    embedding: List[float] = Field(..., description="Normalized dense semantic vector representation")
    response: str = Field(..., description="Cached LLM response")
    token_count: int = Field(default=0, ge=0, description="Tokens consumed by this response")
    model: str = Field(default="gemini-2.5-flash")
    hit_count: int = Field(default=0, ge=0, description="Number of times this cache entry was reused")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CacheLookupResult(BaseModel):
    """Result of a semantic cache lookup request."""
    is_hit: bool
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_query: Optional[str] = None
    response: str
    tokens_saved: int = Field(default=0, ge=0)
    cost_saved_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    source: str = Field(..., description="'CACHE_HIT' or 'LLM_INFERENCE'")


class CacheStats(BaseModel):
    """Aggregated operational metrics, ROI, and latency telemetry."""
    total_queries: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    hit_ratio_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    total_tokens_saved: int = Field(default=0, ge=0)
    total_cost_saved_usd: float = Field(default=0.0, ge=0.0)
    avg_hit_latency_ms: float = Field(default=0.0, ge=0.0)
    avg_miss_latency_ms: float = Field(default=0.0, ge=0.0)
    speedup_factor: float = Field(default=1.0, ge=1.0)
    active_entries_count: int = Field(default=0, ge=0)
