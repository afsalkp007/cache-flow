"""LLM Gateway Proxy with semantic caching, latency telemetry, and ROI savings tracking."""

import os
import time
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Optional, List
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
logger = logging.getLogger(__name__)

from cache_flow.models import CacheLookupResult, CacheStats
from cache_flow.cache import SemanticCache


class BaseLLM(ABC):
    """Abstract interface for LLM completions."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiProvider(BaseLLM):
    """Google Gemini 2.5 Flash completion provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

    def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3},
        }
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API request failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


class MockProvider(BaseLLM):
    """Realistic mock provider simulating LLM latency and contextual generation."""

    def generate(self, prompt: str) -> str:
        # Simulate realistic external API network round-trip (e.g. 500ms)
        time.sleep(0.45)
        p_lower = prompt.lower()

        if "password" in p_lower or "reset" in p_lower:
            return (
                "To reset your account password:\n"
                "1. Navigate to the login portal and click 'Forgot Password'.\n"
                "2. Enter your registered email address to receive a secure 6-digit verification code.\n"
                "3. Follow the link to choose a new password containing at least 12 characters."
            )
        elif "cancel" in p_lower or "subscription" in p_lower or "refund" in p_lower:
            return (
                "To cancel your subscription or request a refund:\n"
                "1. Go to Settings > Billing & Plans.\n"
                "2. Select 'Cancel Subscription'. Your access will remain active until the end of your billing cycle.\n"
                "3. If eligible for a refund under our 14-day policy, an automatic refund will be credited."
            )
        elif "api" in p_lower or "rate limit" in p_lower or "key" in p_lower:
            return (
                "API rate limits are tier-based:\n"
                "- Free Tier: 60 requests/minute\n"
                "- Developer Tier: 600 requests/minute\n"
                "- Enterprise: Custom unmetered access with dedicated egress pools."
            )
        else:
            return (
                f"Thank you for contacting support regarding '{prompt[:50]}...'. "
                "Our team has reviewed your request. You can find comprehensive documentation "
                "in our developer knowledge base or reply here for further assistance."
            )


class LLMGateway:
    """Proxy gateway routing queries through semantic cache with cost/latency telemetry."""

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        provider: Optional[BaseLLM] = None,
        cost_per_1k_tokens: float = 0.00015,
    ):
        self.cache = cache or SemanticCache()
        self.provider = provider or self._resolve_provider()
        self.cost_per_1k_tokens = cost_per_1k_tokens

        # Operational telemetry counters
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens_saved = 0
        self.total_cost_saved = 0.0
        self.hit_latencies: List[float] = []
        self.miss_latencies: List[float] = []

    def _resolve_provider(self) -> BaseLLM:
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
        if provider_name == "gemini" and gemini_key and not gemini_key.startswith("your_"):
            try:
                return GeminiProvider(api_key=gemini_key)
            except Exception as e:
                logger.warning("Failed to initialize GeminiProvider: %s. Using MockProvider.", e)
                return MockProvider()
        return MockProvider()

    def query(self, prompt: str) -> CacheLookupResult:
        """Route user prompt through semantic cache before reaching LLM."""
        clean_prompt = prompt.strip()
        self.total_queries += 1
        start_time = time.time()

        # 1. Check Semantic Cache
        entry, similarity = self.cache.lookup(clean_prompt)

        if entry is not None:
            # CACHE HIT (< 5ms)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            self.cache_hits += 1
            self.hit_latencies.append(elapsed_ms)

            tokens_saved = entry.token_count
            cost_saved = round((tokens_saved / 1000.0) * self.cost_per_1k_tokens, 6)

            self.total_tokens_saved += tokens_saved
            self.total_cost_saved += cost_saved

            return CacheLookupResult(
                is_hit=True,
                similarity=similarity,
                matched_query=entry.query,
                response=entry.response,
                tokens_saved=tokens_saved,
                cost_saved_usd=cost_saved,
                latency_ms=elapsed_ms,
                source="CACHE_HIT",
            )

        # CACHE MISS -> Call LLM Provider
        response_text = self.provider.generate(clean_prompt)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        self.cache_misses += 1
        self.miss_latencies.append(elapsed_ms)

        # Approximate token count (1 word ~= 1.3 tokens)
        token_count = max(20, int(len(response_text.split()) * 1.3))

        # Store generation in semantic cache
        self.cache.insert(
            query=clean_prompt,
            response=response_text,
            token_count=token_count,
            model="gemini-2.5-flash",
        )

        return CacheLookupResult(
            is_hit=False,
            similarity=similarity,
            matched_query=None,
            response=response_text,
            tokens_saved=0,
            cost_saved_usd=0.0,
            latency_ms=elapsed_ms,
            source="LLM_INFERENCE",
        )

    def get_stats(self) -> CacheStats:
        """Calculate live hit ratio, ROI, and speedup multipliers."""
        hit_ratio = (
            round((self.cache_hits / self.total_queries) * 100, 1)
            if self.total_queries > 0
            else 0.0
        )
        avg_hit_lat = (
            round(sum(self.hit_latencies) / len(self.hit_latencies), 2)
            if self.hit_latencies
            else 2.0
        )
        avg_miss_lat = (
            round(sum(self.miss_latencies) / len(self.miss_latencies), 2)
            if self.miss_latencies
            else 500.0
        )
        speedup = (
            round(avg_miss_lat / max(avg_hit_lat, 0.1), 1)
            if avg_hit_lat > 0
            else 1.0
        )

        return CacheStats(
            total_queries=self.total_queries,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            hit_ratio_percent=hit_ratio,
            total_tokens_saved=self.total_tokens_saved,
            total_cost_saved_usd=round(self.total_cost_saved, 6),
            avg_hit_latency_ms=avg_hit_lat,
            avg_miss_latency_ms=avg_miss_lat,
            speedup_factor=speedup,
            active_entries_count=self.cache.size(),
        )

    def reset_stats(self) -> None:
        """Reset operational telemetry counters."""
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens_saved = 0
        self.total_cost_saved = 0.0
        self.hit_latencies.clear()
        self.miss_latencies.clear()
