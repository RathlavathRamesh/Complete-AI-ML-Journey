import streamlit as st
from app_core import answer_question

# Used for local API testing only. Streamlit deployment uses app_core.py

st.set_page_config(
    page_title="Enterprise Policy RAG Assistant",
    page_icon="📘",
    layout="wide"
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align:center;'>📘 Enterprise Policy RAG Assistant</h1>
    <p style='text-align:center; color:gray;'>
    Trustworthy policy question answering with evidence, metrics, and confidence.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 🧠 System Overview")
    st.markdown(
        """
        **Enterprise-grade RAG system**
        
        - 🔍 Dense retrieval (FAISS)
        - 🎯 Cross-encoder re-ranking
        - 🧠 Grounded LLM generation
        - 🧪 Semantic faithfulness evaluation
        - 📊 Transparent metrics & insights
        
        Built for **policy, HR, and compliance** use cases.
        """
    )

    st.divider()
    show_sources = st.checkbox("Show Evidence Sources", True)
    show_metrics = st.checkbox("Show Detailed Metrics", False)

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= INPUT =================
query = st.chat_input("Ask a policy-related question…")

if query:
    with st.spinner("🔎 Retrieving evidence and generating answer..."):
        data = answer_question(query)


    answer = data["answer"]
    sources = data["sources"]
    metrics = data["metrics"]
    insight = data["insight"]

    # ================= KPI BAR =================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("⏱ Retrieval (ms)", metrics["retrieval_time_ms"])
    col2.metric("🎯 Re-rank (ms)", metrics["rerank_time_ms"])
    col3.metric("🧠 Generation (ms)", metrics["generation_time_ms"])
    col4.metric("🧪 Faithfulness", metrics["faithfulness_score"])

    # ================= CONFIDENCE BADGE =================
    if metrics["faithfulness_score"] >= 0.6:
        st.success("🟢 High Confidence Answer")
    elif metrics["faithfulness_score"] >= 0.35:
        st.warning("🟡 Medium Confidence Answer")
    else:
        st.error("🔴 Low Confidence – Verify Manually")

    st.divider()

    # ================= ANSWER =================
    st.markdown("## 🧠 Answer")
    st.markdown(answer)

    # ================= INSIGHTS =================
    st.markdown("## 🔍 System Insights")
    st.markdown(insight)

    # ================= DETAILS =================
    if show_sources or show_metrics:
        st.divider()
        left, right = st.columns(2)

        if show_sources:
            with left:
                with st.expander("📚 Evidence & Citations", expanded=False):
                    st.markdown(sources)

        if show_metrics:
            with right:
                with st.expander("📊 Detailed Metrics", expanded=False):
                    st.json(metrics)
