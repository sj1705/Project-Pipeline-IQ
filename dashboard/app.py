import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8003"

st.set_page_config(page_title="PipelineIQ Dashboard", layout="wide")
st.title("🔧 PipelineIQ — Self-Optimizing RAG Dashboard")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Pipeline Health",
    "Latency Analysis",
    "Cost Analysis",
    "Retrieval Quality",
    "Optimizer",
])


def fetch_metrics(limit=50):
    """Fetch recent query logs from API."""
    try:
        resp = requests.get(f"{API_URL}/metrics?limit={limit}")
        return resp.json()
    except:
        return None


# =====================
# PAGE 1: Pipeline Health
# =====================
if page == "Pipeline Health":
    st.header("📊 Pipeline Health Overview")

    # Health check
    try:
        health = requests.get(f"{API_URL}/health").json()
        col1, col2 = st.columns(2)
        col1.metric("API Status", "🟢 Online" if health["status"] == "ok" else "🔴 Down")
        col2.metric("Database", "🟢 " + health["database"])
    except:
        st.error("❌ Cannot connect to API. Is the server running?")
        st.stop()

    st.divider()

    # Key metrics
    data = fetch_metrics(20)
    if data and data["recent_queries"]:
        recent = data["recent_queries"]
        total = data["total_queries"]

        # Calculate averages
        avg_latency = sum(q["latency_ms"] or 0 for q in recent) / len(recent)
        avg_cost = sum(q["cost"] or 0 for q in recent) / len(recent)
        faithfulness_scores = [
            q["evaluation_scores"]["faithfulness"]
            for q in recent
            if q.get("evaluation_scores") and "faithfulness" in q["evaluation_scores"]
        ]
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", total)
        col2.metric("Avg Faithfulness", f"{avg_faithfulness:.2f}",
                   delta="Good" if avg_faithfulness >= 0.7 else "⚠️ Low")
        col3.metric("Avg Latency", f"{avg_latency:.0f}ms",
                   delta="Fast" if avg_latency < 5000 else "⚠️ Slow")
        col4.metric("Avg Cost", f"${avg_cost:.4f}")

        st.divider()

        # Model distribution
        st.subheader("Model Usage")
        model_counts = {}
        for q in recent:
            model = q["model_used"] or "unknown"
            model_counts[model] = model_counts.get(model, 0) + 1
        st.bar_chart(pd.Series(model_counts))

    else:
        st.info("No queries yet. Use /query/agent to start logging data.")

    # A/B test status
    st.divider()
    st.subheader("A/B Test Status")
    try:
        ab = requests.get(f"{API_URL}/ab-test").json()
        if ab.get("active"):
            st.warning(f"🧪 A/B Test Active: {ab.get('ab_status', '')}")
        else:
            st.success("✅ No active A/B test")
        with st.expander("Current Config"):
            st.json(ab.get("use_config", {}))
    except:
        st.error("Cannot fetch A/B test status")


# =====================
# PAGE 2: Latency Analysis
# =====================
elif page == "Latency Analysis":
    st.header("⏱️ Latency Analysis")

    data = fetch_metrics(50)
    if data and data["recent_queries"]:
        df = pd.DataFrame(data["recent_queries"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")

        # Latency over time
        st.subheader("Latency Over Time (ms)")
        st.line_chart(df.set_index("created_at")["latency_ms"])

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Min Latency", f"{df['latency_ms'].min():.0f}ms")
        col2.metric("Avg Latency", f"{df['latency_ms'].mean():.0f}ms")
        col3.metric("Max Latency", f"{df['latency_ms'].max():.0f}ms")

        # By model
        st.subheader("Latency by Model")
        model_latency = df.groupby("model_used")["latency_ms"].mean()
        st.bar_chart(model_latency)

    else:
        st.info("No data yet.")


# =====================
# PAGE 3: Cost Analysis
# =====================
elif page == "Cost Analysis":
    st.header("💰 Cost Analysis")

    data = fetch_metrics(50)
    if data and data["recent_queries"]:
        df = pd.DataFrame(data["recent_queries"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")

        # Cost over time
        st.subheader("Cost Per Query Over Time ($)")
        st.line_chart(df.set_index("created_at")["cost"])

        # Cumulative cost
        df["cumulative_cost"] = df["cost"].cumsum()
        st.subheader("Cumulative Cost ($)")
        st.line_chart(df.set_index("created_at")["cumulative_cost"])

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spend", f"${df['cost'].sum():.4f}")
        col2.metric("Avg Per Query", f"${df['cost'].mean():.4f}")
        col3.metric("Total Tokens", f"{df['token_count'].sum():,}")

        # Cost by model
        st.subheader("Cost by Model")
        model_cost = df.groupby("model_used")["cost"].sum()
        st.bar_chart(model_cost)

    else:
        st.info("No data yet.")


# =====================
# PAGE 4: Retrieval Quality
# =====================
elif page == "Retrieval Quality":
    st.header("🎯 Retrieval Quality")

    data = fetch_metrics(50)
    if data and data["recent_queries"]:
        df = pd.DataFrame(data["recent_queries"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")

        # Extract evaluation scores
        faithfulness = []
        relevancy = []
        for _, row in df.iterrows():
            scores = row.get("evaluation_scores") or {}
            faithfulness.append(scores.get("faithfulness", None))
            relevancy.append(scores.get("answer_relevancy", None))

        df["faithfulness"] = faithfulness
        df["relevancy"] = relevancy

        # Faithfulness over time
        st.subheader("Faithfulness Over Time")
        faith_df = df.dropna(subset=["faithfulness"])
        if not faith_df.empty:
            st.line_chart(faith_df.set_index("created_at")["faithfulness"])
            st.caption("Threshold: 0.7 (below = unreliable answers)")

        # Relevancy over time
        st.subheader("Answer Relevancy Over Time")
        rel_df = df.dropna(subset=["relevancy"])
        if not rel_df.empty:
            st.line_chart(rel_df.set_index("created_at")["relevancy"])

        # Stats
        col1, col2 = st.columns(2)
        if not faith_df.empty:
            col1.metric("Avg Faithfulness", f"{faith_df['faithfulness'].mean():.3f}")
        if not rel_df.empty:
            col2.metric("Avg Relevancy", f"{rel_df['relevancy'].mean():.3f}")

    else:
        st.info("No data yet.")


# =====================
# PAGE 5: Optimizer
# =====================
elif page == "Optimizer":
    st.header("🧠 Optimization Agent")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Config")
        try:
            ab = requests.get(f"{API_URL}/ab-test").json()
            st.json(ab.get("use_config", {}))
            st.caption(f"Version: {ab.get('config_version', 'default')}")
        except:
            st.error("Cannot fetch config")

    with col2:
        st.subheader("Run Optimizer")
        window = st.slider("Analysis window (queries)", 5, 50, 20)
        if st.button("🚀 Run Optimizer Now"):
            with st.spinner("Optimizer analyzing performance..."):
                try:
                    result = requests.get(f"{API_URL}/optimize?window={window}").json()
                    if result.get("status") == "optimization_proposed":
                        st.success("✅ New config proposed!")
                    elif result.get("status") == "no_changes_needed":
                        st.info("Pipeline is performing well. No changes needed.")
                    else:
                        st.warning(result.get("status", "Unknown"))
                    st.json(result)
                except Exception as e:
                    st.error(f"Optimizer failed: {e}")