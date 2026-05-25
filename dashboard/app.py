import streamlit as st
import requests

API_URL = "http://127.0.0.1:8003"

st.set_page_config(page_title="PipelineIQ Dashboard", layout="wide")
st.title("🔧 PipelineIQ — Self-Optimizing RAG Dashboard")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Pipeline Health", "Query Metrics", "Optimizer"])


# --- Pipeline Health Page ---
if page == "Pipeline Health":
    st.header("📊 Pipeline Health")

    # Fetch health
    try:
        health = requests.get(f"{API_URL}/health").json()
        col1, col2 = st.columns(2)
        col1.metric("Status", health["status"])
        col2.metric("Database", health["database"])
    except Exception as e:
        st.error(f"Cannot connect to API: {e}")

    st.divider()

    # Fetch metrics
    try:
        metrics = requests.get(f"{API_URL}/metrics?limit=20").json()
        st.subheader(f"Total Queries: {metrics['total_queries']}")

        if metrics["recent_queries"]:
            # Calculate averages
            recent = metrics["recent_queries"]
            avg_latency = sum(q["latency_ms"] or 0 for q in recent) / len(recent)
            avg_cost = sum(q["cost"] or 0 for q in recent) / len(recent)

            faithfulness_scores = []
            for q in recent:
                if q["evaluation_scores"] and "faithfulness" in q["evaluation_scores"]:
                    faithfulness_scores.append(q["evaluation_scores"]["faithfulness"])
            avg_faithfulness = (
                sum(faithfulness_scores) / len(faithfulness_scores)
                if faithfulness_scores else 0
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Faithfulness", f"{avg_faithfulness:.2f}",
                       delta="Good" if avg_faithfulness > 0.7 else "Low")
            col2.metric("Avg Latency", f"{avg_latency:.0f} ms",
                       delta="Fast" if avg_latency < 5000 else "Slow")
            col3.metric("Avg Cost", f"${avg_cost:.4f}/query")

    except Exception as e:
        st.error(f"Cannot fetch metrics: {e}")

    st.divider()

    # Fetch A/B test status
    try:
        ab_status = requests.get(f"{API_URL}/ab-test").json()
        st.subheader("A/B Test Status")
        if ab_status.get("active"):
            st.info(f"🧪 A/B Test Active: {ab_status.get('ab_status', '')}")
        else:
            st.success("✅ No A/B test running")

        st.json(ab_status.get("use_config", {}))
    except Exception as e:
        st.error(f"Cannot fetch A/B test status: {e}")


# --- Query Metrics Page ---
elif page == "Query Metrics":
    st.header("📈 Recent Queries")

    try:
        metrics = requests.get(f"{API_URL}/metrics?limit=20").json()

        if metrics["recent_queries"]:
            import pandas as pd
            df = pd.DataFrame(metrics["recent_queries"])
            df["created_at"] = pd.to_datetime(df["created_at"])

            # Table view
            st.dataframe(
                df[["query", "model_used", "latency_ms", "cost", "created_at"]],
                use_container_width=True,
            )

            # Charts
            st.subheader("Latency Over Time")
            st.line_chart(df.set_index("created_at")["latency_ms"])

            st.subheader("Cost Over Time")
            st.line_chart(df.set_index("created_at")["cost"])
        else:
            st.info("No queries logged yet.")

    except Exception as e:
        st.error(f"Cannot fetch metrics: {e}")


# --- Optimizer Page ---
elif page == "Optimizer":
    st.header("🧠 Optimization Agent")

    st.write("Trigger the optimizer to analyze recent performance and propose config changes.")

    if st.button("🚀 Run Optimizer"):
        with st.spinner("Optimizer is analyzing..."):
            try:
                result = requests.get(f"{API_URL}/optimize").json()
                st.json(result)
            except Exception as e:
                st.error(f"Optimizer failed: {e}")

    st.divider()

    # Show current config
    st.subheader("Current Pipeline Config")
    try:
        ab_status = requests.get(f"{API_URL}/ab-test").json()
        st.json(ab_status.get("use_config", {}))
        st.caption(f"Config version: {ab_status.get('config_version', 'default')}")
    except Exception as e:
        st.error(f"Cannot fetch config: {e}")