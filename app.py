"""
SegmentIQ — Streamlit experience
Chat-first banking analytics agent with live trace and visual outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from agent_core import SegmentIQAgent
from data_store import store

st.set_page_config(page_title="SegmentIQ", page_icon="🏦", layout="wide")

CUSTOM_CSS = """
<style>
    .main { background: linear-gradient(180deg, #f7fbff 0%, #eef4f8 100%); }
    .block-container { padding-top: 1.5rem; }
    .segmentiq-header {
        background: linear-gradient(120deg, #0b1f3a 0%, #123f67 55%, #1f7a8c 100%);
        color: white; padding: 1.4rem 1.6rem; border-radius: 16px;
        margin-bottom: 1rem; box-shadow: 0 10px 30px rgba(11,31,58,0.18);
    }
    .segmentiq-header h1 { margin: 0; font-size: 2rem; }
    .segmentiq-header p { margin: 0.35rem 0 0; opacity: 0.92; }
    .metric-card {
        background: white; border: 1px solid #dbe7f0; border-radius: 14px;
        padding: 0.9rem 1rem; box-shadow: 0 4px 14px rgba(18,63,103,0.06);
    }
    .chat-user {
        background: #e8f3ff; border-left: 4px solid #1f7a8c;
        padding: 0.8rem 1rem; border-radius: 12px; margin: 0.5rem 0;
    }
    .chat-agent {
        background: white; border-left: 4px solid #0b1f3a;
        padding: 0.8rem 1rem; border-radius: 12px; margin: 0.5rem 0;
        border: 1px solid #dbe7f0;
    }
    .trace-box {
        background: #0b1f3a; color: #d7e7ff; padding: 1rem;
        border-radius: 12px; font-family: monospace; font-size: 0.85rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="segmentiq-header">
        <h1>SegmentIQ</h1>
        <p>Agentic customer segmentation and personalization for retail banking</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            store.load_csv_bytes(uploaded.getvalue(), uploaded.name)
            st.success(f"Loaded {uploaded.name}")
        except Exception as exc:
            st.error(str(exc))
    elif not store.is_loaded:
        store.auto_load()

    info = store.summary()
    if info.get("loaded"):
        st.markdown(
            f"<div class='metric-card'>Rows: <b>{info['rows']:,}</b><br>"
            f"Customers: <b>{info.get('unique_customers', '—')}</b><br>"
            f"Source: <b>{Path(str(info['source'])).name}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Add data/customers.csv or upload a file.")

    st.divider()
    st.markdown("**Quick prompts**")
    samples = [
        "Segment customers into priority, regular and dormant based on balance and transaction frequency",
        "On what basis were priority customers selected?",
        "What is the average transaction size for priority and regular customers?",
        "Which regular customers can be converted to priority customers?",
    ]
    for sample in samples:
        if st.button(sample, use_container_width=True):
            st.session_state.pending_query = sample

if not store.is_loaded:
    st.info("Upload or place your dataset to begin.")
    st.stop()

left, center, right = st.columns([1.1, 1.8, 1.1])

with left:
    st.subheader("Segment snapshot")
    if store.has_segments():
        seg_df = store.get_segments()
        counts = seg_df["segment"].value_counts().reset_index()
        counts.columns = ["segment", "count"]
        fig = px.pie(counts, names="segment", values="count", hole=0.45, color="segment")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        if store.last_export_path:
            with open(store.last_export_path, "rb") as handle:
                st.download_button(
                    "Download segment CSV",
                    data=handle,
                    file_name=Path(store.last_export_path).name,
                    use_container_width=True,
                )
    else:
        st.caption("Run a segmentation query to populate this panel.")

with center:
    st.subheader("Analytics chat")
    for msg in st.session_state.messages:
        css = "chat-user" if msg["role"] == "user" else "chat-agent"
        st.markdown(f"<div class='{css}'>{msg['content']}</div>", unsafe_allow_html=True)

    default_query = st.session_state.pop("pending_query", "")
    query = st.text_area("Ask SegmentIQ", value=default_query, height=90)
    run_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if run_clicked and query.strip():
        st.session_state.messages.append({"role": "user", "content": query.strip()})
        with st.spinner("SegmentIQ is thinking..."):
            try:
                result = SegmentIQAgent().run(query.strip())
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        if result.needs_clarification:
            answer = result.clarifying_question or "Could you clarify?"
        else:
            answer = result.summary or "Analysis completed."

        st.session_state.messages.append({"role": "assistant", "content": answer.replace("\n", "<br>")})
        st.session_state.last_result = result
        st.rerun()

with right:
    st.subheader("Reasoning trace")
    last = st.session_state.get("last_result")
    if last:
        st.markdown(
            f"<div class='trace-box'>{last.trace_as_text().replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Structured output"):
            st.json(last.tools_used or last.tool_output)
    else:
        st.caption("Tool decisions will appear here after your first query.")

if store.has_segments():
    st.divider()
    st.subheader("Segment personas")
    personas = store.segment_meta.get("personas", [])
    cols = st.columns(min(len(personas), 3) or 1)
    for idx, persona in enumerate(personas):
        with cols[idx % len(cols)]:
            st.markdown(f"### {persona.get('title', persona.get('segment'))}")
            st.write(persona.get("tagline", ""))
            for insight in persona.get("insights", []):
                st.write(f"- {insight}")
