"""Interactive Streamlit Semantic Caching & Cost Optimizer Dashboard."""

import os
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

import streamlit as st

from cache_flow.cache import SemanticCache
from cache_flow.gateway import LLMGateway, MockProvider, GeminiProvider

# Streamlit Page Config
st.set_page_config(
    page_title="CacheFlow | Semantic Caching & LLM Cost Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-hit {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
    }
    .badge-miss {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">⚡ CacheFlow: Semantic Caching & LLM Cost Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Sub-millisecond semantic vector caching, intelligent query deduplication, and real-time LLM cost analytics.</div>',
    unsafe_allow_html=True,
)

# Initialize Session State
if "cache" not in st.session_state:
    st.session_state["cache"] = SemanticCache(similarity_threshold=0.70, max_entries=500)

if "gateway" not in st.session_state:
    st.session_state["gateway"] = LLMGateway(cache=st.session_state["cache"])

cache: SemanticCache = st.session_state["cache"]
gateway: LLMGateway = st.session_state["gateway"]

# Sidebar Settings
st.sidebar.header("⚙️ Cache Configuration")

threshold = st.sidebar.slider(
    "Cosine Similarity Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
    help="Minimum vector similarity required to trigger a cache hit instead of calling the LLM.",
)
cache.similarity_threshold = threshold

provider_choice = st.sidebar.selectbox(
    "Underlying LLM",
    options=["gemini", "mock"],
    index=0 if os.getenv("GEMINI_API_KEY") and not os.getenv("GEMINI_API_KEY").startswith("your_") else 1,
    help="Select 'gemini' for live Gemini API calls or 'mock' for zero-cost simulated latency.",
)

if provider_choice == "gemini" and os.getenv("GEMINI_API_KEY"):
    gateway.provider = GeminiProvider(api_key=os.getenv("GEMINI_API_KEY"))
else:
    gateway.provider = MockProvider()

st.sidebar.divider()
st.sidebar.subheader("🧪 Paired Demonstration Presets")

DEMO_PAIRS = {
    "🔑 Password Reset (Paraphrased Query)": [
        "How do I reset my account password?",
        "What are the steps to reset password?",
    ],
    "💳 Subscription & Refund (Different Wording)": [
        "I want to cancel my subscription and get a refund.",
        "How can I cancel subscription and request refund?",
    ],
    "⚡ API Rate Limits (Synonym Match)": [
        "What are the API rate limits for developers?",
        "How many API requests can I make per minute?",
    ],
}

selected_demo = st.sidebar.selectbox("Select Test Pair", list(DEMO_PAIRS.keys()))
pair_queries = DEMO_PAIRS[selected_demo]

st.sidebar.caption("💡 **Tip:** Execute Query #1 first (generates Cache Miss & populates vector index), then run Query #2 to observe the instant sub-millisecond Cache Hit!")

if st.sidebar.button("🧹 Clear Cache & Reset Metrics", use_container_width=True):
    cache.clear()
    gateway.reset_stats()
    st.success("Cache and metrics reset!")
    st.rerun()

# Top KPI Telemetry Row
stats = gateway.get_stats()
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Hit Ratio", f"{stats.hit_ratio_percent}%", delta=f"{stats.cache_hits} hits / {stats.total_queries} total")
with k2:
    st.metric("Speedup Multiplier", f"{stats.speedup_factor}x", delta=f"~{stats.avg_hit_latency_ms}ms vs ~{stats.avg_miss_latency_ms}ms")
with k3:
    st.metric("Tokens Conserved", f"{stats.total_tokens_saved:,}", delta=f"{stats.active_entries_count} entries stored")
with k4:
    st.metric("Cost Saved", f"${stats.total_cost_saved_usd:.6f}")
with k5:
    st.metric("Active Cache Entries", stats.active_entries_count)

st.markdown("---")

# Query Playground
st.subheader("💬 Query Execution Playground")

col_q1, col_q2 = st.columns(2)
with col_q1:
    if st.button(f"1️⃣ Load Seed Query: '{pair_queries[0]}'", use_container_width=True):
        st.session_state["active_prompt"] = pair_queries[0]
with col_q2:
    if st.button(f"2️⃣ Load Paraphrased Query: '{pair_queries[1]}'", use_container_width=True):
        st.session_state["active_prompt"] = pair_queries[1]

current_prompt = st.text_area(
    "Enter query:",
    value=st.session_state.get("active_prompt", pair_queries[0]),
    height=80,
)

if st.button("🚀 Send Query via CacheFlow Gateway", type="primary", use_container_width=True):
    with st.spinner("Processing through semantic cache gateway..."):
        result = gateway.query(current_prompt)
        st.session_state["last_result"] = result

# Display Last Result
if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.markdown("### 🔍 Gateway Telemetry")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        if res.is_hit:
            st.success(f"**⚡ CACHE HIT** (Similarity: `{res.similarity:.3f}`)")
        else:
            st.warning(f"**🐢 CACHE MISS** (Similarity: `{res.similarity:.3f}`)")
    with col_r2:
        st.metric("Latency", f"{res.latency_ms:.2f} ms")
    with col_r3:
        st.metric("Tokens Saved", res.tokens_saved)
    with col_r4:
        st.metric("Cost Saved", f"${res.cost_saved_usd:.6f}")

    if res.matched_query:
        st.info(f"**Semantically Matched Prior Query:** *\"{res.matched_query}\"*")

    st.markdown("**Gateway Output:**")
    st.code(res.response, language="text")

# Active Cache Table
st.markdown("---")
st.subheader("📚 In-Memory Semantic Vector Store")

entries = cache.get_entries()
if entries:
    table_data = [
        {
            "Hash Key": e.key_id,
            "Cached Query": e.query,
            "Hits": e.hit_count,
            "Tokens": e.token_count,
            "Model": e.model,
            "Last Accessed": e.last_accessed_at[11:19],
        }
        for e in entries
    ]
    st.dataframe(table_data, use_container_width=True)
else:
    st.info("The semantic cache is currently empty. Run a query above to seed the vector store.")
