"""
SegmentIQ - Streamlit experience.
Chat-first banking analytics agent with live trace and visual outputs.
"""

from __future__ import annotations

import html
from pathlib import Path

import plotly.express as px
import streamlit as st

from agent_core import SegmentIQAgent
from data_store import store

st.set_page_config(page_title="SegmentIQ", page_icon=":bank:", layout="wide")

CUSTOM_CSS = """
<style>
    :root {
        --bg: #f6f4ef;
        --panel: #fffdfa;
        --ink: #181513;
        --muted: #655b52;
        --border: #d7cec2;
        --accent: #0f766e;
        --accent-strong: #115e59;
        --accent-soft: #dff3f0;
        --shadow: 0 14px 36px rgba(24, 21, 19, 0.08);
        --radius-lg: 20px;
        --radius-md: 14px;
    }

    [data-testid="stAppViewContainer"] .main {
        background:
            radial-gradient(circle at top left, #f3ece4 0, transparent 30%),
            linear-gradient(180deg, #f8f5ef 0%, #f3efe7 100%);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    .segmentiq-shell {
        background: rgba(255, 253, 250, 0.84);
        border: 1px solid rgba(215, 206, 194, 0.9);
        border-radius: 28px;
        padding: 1.35rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .segmentiq-header {
        background:
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.24), transparent 35%),
            linear-gradient(135deg, #123f3b 0%, #0f766e 55%, #1a8b83 100%);
        color: #ffffff;
        border-radius: 24px;
        padding: 1.6rem 1.7rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 118, 110, 0.22);
    }

    .segmentiq-header h1 {
        margin: 0;
        font-size: 2.1rem;
        line-height: 1.1;
        color: #ffffff;
    }

    .segmentiq-header p {
        margin: 0.45rem 0 0;
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.98rem;
    }

    .section-title {
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .metric-card {
        background: linear-gradient(180deg, #fffdfa 0%, #f7f3ec 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 0.95rem 1rem;
        color: var(--ink);
        margin-bottom: 0.8rem;
    }

    .metric-card strong {
        display: block;
        font-size: 1.35rem;
        color: var(--ink);
        margin-top: 0.15rem;
    }

    .metric-card,
    .metric-card * {
        color: var(--ink) !important;
    }

    .mini-meta {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 0.15rem;
    }

    .persona-card {
        background: #fffdfa;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem;
        min-height: 100%;
    }

    .persona-card h3 {
        margin-top: 0;
        margin-bottom: 0.35rem;
        color: var(--ink);
    }

    .persona-tagline {
        color: var(--muted);
        margin-bottom: 0.7rem;
    }

    .trace-box {
        background: #171717;
        color: #f5f5f5;
        border-radius: 16px;
        padding: 1rem;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.84rem;
        line-height: 1.5;
        white-space: pre-wrap;
        min-height: 260px;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top, rgba(255, 255, 255, 0.08), transparent 28%),
            linear-gradient(180deg, #102927 0%, #0b1c1b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f8f5ef !important;
    }

    section[data-testid="stSidebar"] button {
        background: #ecfdf5 !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] button *,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #0f172a !important;
    }

    section[data-testid="stSidebar"] button:hover {
        background: #d1fae5 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px dashed rgba(255, 255, 255, 0.28) !important;
    }

    .stTextArea textarea {
        border-radius: 16px !important;
        border: 1px solid var(--border) !important;
        background: #fffdfa !important;
        color: var(--ink) !important;
        min-height: 120px !important;
    }

    .stTextArea label,
    .stTextArea label p,
    .stTextArea textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    .main .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 22px rgba(15, 118, 110, 0.18);
    }

    .main .stButton > button:hover {
        filter: brightness(1.03);
    }

    .main .stButton > button *,
    .main [data-testid="stDownloadButton"] button * {
        color: inherit !important;
    }

    .main [data-testid="stDownloadButton"] button {
        background: #fffdfa !important;
        color: var(--ink) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    .main [data-testid="stAlert"] {
        border-radius: 16px !important;
        border: 1px solid rgba(15, 118, 110, 0.12) !important;
    }

    .main [data-testid="stAlert"] *,
    .main [data-testid="stCaptionContainer"],
    .main [data-testid="stCaptionContainer"] * {
        color: var(--ink) !important;
    }

    .main [data-testid="stInfo"] {
        background: #e7f0fb !important;
    }

    .main [data-testid="stNotificationContentInfo"] *,
    .main [data-testid="stNotificationContentWarning"] *,
    .main [data-testid="stNotificationContentError"] *,
    .main [data-testid="stNotificationContentSuccess"] * {
        color: var(--ink) !important;
    }

    .main [data-testid="stMarkdownContainer"],
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li,
    .main label,
    .main .st-emotion-cache-ue6h4q,
    .main .st-emotion-cache-16idsys {
        color: var(--ink);
    }

    [data-testid="column"] > div {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1rem 1rem 0.85rem;
        box-shadow: var(--shadow);
        height: 100%;
    }

    [data-testid="stChatMessage"] {
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(255, 253, 250, 0.92);
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: var(--ink);
    }
</style>
"""


def render_header() -> None:
    st.markdown(
        """
        <div class="segmentiq-shell">
            <div class="segmentiq-header">
                <h1>SegmentIQ</h1>
                <p>Agentic customer segmentation and personalization for retail banking</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_card(info: dict[str, object]) -> None:
    source_name = Path(str(info["source"])).name if info.get("source") else "Unknown"
    customer_count = info.get("unique_customers")
    customer_text = f"{customer_count:,}" if isinstance(customer_count, int) else "N/A"

    st.markdown(
        f"""
        <div class="metric-card">
            <div>Rows</div>
            <strong>{info['rows']:,}</strong>
            <div class="mini-meta">Customers: {customer_text}</div>
            <div class="mini-meta">Source: {html.escape(source_name)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
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
            render_dataset_card(info)
        else:
            st.warning("Add `data/customers.csv` or upload a file to begin.")

        st.divider()
        st.markdown("**Quick prompts**")
        samples = [
            "Segment customers into priority, regular and dormant based on balance and transaction frequency",
            "On what basis were priority customers selected?",
            "What is the average transaction size for priority and regular customers?",
            "Which regular customers can be converted to priority customers?",
        ]
        for index, sample in enumerate(samples):
            if st.button(sample, key=f"sample_{index}", use_container_width=True):
                st.session_state.pending_query = sample


def render_snapshot_panel() -> None:
    st.markdown("<div class='section-title'>Segment Snapshot</div>", unsafe_allow_html=True)
    if not store.has_segments():
        st.info("Run a segmentation query to populate this panel.")
        return

    seg_df = store.get_segments()
    counts = seg_df["segment"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    fig = px.pie(
        counts,
        names="segment",
        values="count",
        hole=0.48,
        color="segment",
        color_discrete_sequence=["#0f766e", "#d97706", "#8b5cf6", "#ef4444", "#2563eb"],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=290,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#181513"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"{len(seg_df):,} customer records are currently assigned to segments.")

    if store.last_export_path:
        with open(store.last_export_path, "rb") as handle:
            st.download_button(
                "Download segment CSV",
                data=handle.read(),
                file_name=Path(store.last_export_path).name,
                use_container_width=True,
            )


def render_chat_panel() -> None:
    st.markdown("<div class='section-title'>Analytics Chat</div>", unsafe_allow_html=True)

    if st.session_state.messages:
        for msg in st.session_state.messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            with st.chat_message(role):
                st.markdown(msg["content"])
    else:
        st.info("Ask a segmentation or customer analytics question to get started.")

    default_query = st.session_state.pop("pending_query", "")
    with st.form("segmentiq-query-form", clear_on_submit=False):
        query = st.text_area(
            "Ask SegmentIQ",
            value=default_query,
            placeholder="Example: Compare dormant customers against regular customers and suggest a reactivation strategy.",
        )
        run_clicked = st.form_submit_button("Analyze", use_container_width=True)

    if not run_clicked:
        return

    if not query.strip():
        st.warning("Enter a question before running the analysis.")
        return

    st.session_state.messages.append({"role": "user", "content": query.strip()})

    with st.spinner("SegmentIQ is thinking..."):
        try:
            result = SegmentIQAgent().run(query.strip())
        except RuntimeError as exc:
            st.error(str(exc))
            return

    answer = result.clarifying_question if result.needs_clarification else result.summary
    st.session_state.messages.append(
        {"role": "assistant", "content": answer or "Analysis completed."}
    )
    st.session_state.last_result = result
    st.rerun()


def render_trace_panel() -> None:
    st.markdown("<div class='section-title'>Reasoning Trace</div>", unsafe_allow_html=True)
    last = st.session_state.get("last_result")
    if not last:
        st.info("Tool decisions and intermediate reasoning will appear here after your first query.")
        return

    st.markdown(
        f"<div class='trace-box'>{html.escape(last.trace_as_text())}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Structured output"):
        st.json(last.tools_used or last.tool_output)


def render_personas() -> None:
    if not store.has_segments():
        return

    personas = store.segment_meta.get("personas", [])
    if not personas:
        return

    st.divider()
    st.markdown("<div class='section-title'>Segment Personas</div>", unsafe_allow_html=True)
    cols = st.columns(min(len(personas), 3))
    for idx, persona in enumerate(personas):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="persona-card">
                    <h3>{html.escape(str(persona.get('title', persona.get('segment', 'Segment'))))}</h3>
                    <div class="persona-tagline">{html.escape(str(persona.get('tagline', '')))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for insight in persona.get("insights", []):
                st.write(f"- {insight}")


if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_header()
render_sidebar()

if not store.is_loaded:
    st.info("Upload a dataset or place `customers.csv` in the `data` folder to begin.")
    st.stop()

left, center, right = st.columns([1.05, 1.8, 1.15], gap="large")

with left:
    render_snapshot_panel()

with center:
    render_chat_panel()

with right:
    render_trace_panel()

render_personas()