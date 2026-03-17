import streamlit as st
from agentic.multi_agent import run_multi_agent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import builtins

load_dotenv()

st.set_page_config(
    page_title="QueryMind",
    page_icon="⚡",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "tools_stats"   not in st.session_state:
    st.session_state.tools_stats = {
        "search_document": 0,
        "search_web":      0,
        "calculate":       0,
        "summarize":       0
    }

TOOL_ICONS = {
    "search_document": "📄",
    "search_web":      "🌐",
    "calculate":       "🧮",
    "summarize":       "📝",
}

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚡ QueryMind")
    st.caption("multi-tool agent v2.0")
    st.divider()

    # Stats
    st.subheader("📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    c2.metric("Memory",  len(st.session_state.chat_history) // 2)
    st.divider()

    # Tools
    st.subheader("🔧 Tools Usage")
    for tool, count in st.session_state.tools_stats.items():
        icon = TOOL_ICONS.get(tool, "🔧")
        st.write(f"{icon} **{tool}** — {count}×")
    st.divider()

    # Knowledge base
    st.subheader("📁 Knowledge Base")
    st.info("📄 document.pdf\n\n6 pages · 13 chunks")
    st.divider()

    # Clear
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.chat_history  = []
        st.session_state.total_queries = 0
        st.session_state.tools_stats   = {k: 0 for k in st.session_state.tools_stats}
        st.rerun()

    st.caption("created by **rahouba**")

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
st.title("⚡ QueryMind")
st.caption("Ask me anything. I pick the right tool automatically.")
st.divider()

# Welcome
if not st.session_state.messages:
    st.info("👋 Start by asking a question below.")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.success("📄 Quels sont les types d'index ?")
        st.success("🧮 Gain : 150ms vs 5ms ?")
    with c2:
        st.success("🌐 Qu'est-ce que PostgreSQL ?")
        st.success("📝 Résume le cours")

# Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("tools_used"):
            tools_str = " · ".join(
                f"{TOOL_ICONS.get(t,'🔧')} {t}"
                for t in msg["tools_used"]
            )
            st.caption(f"Tools used: {tools_str}")

# ── Chat input ────────────────────────────────────────────
question = st.chat_input("Message QueryMind...")

if question:
    st.session_state.total_queries += 1

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            _p = builtins.print
            builtins.print = lambda *a, **k: _p(*a, **k)
            try:
                result = run_multi_agent(question, st.session_state.chat_history)
            finally:
                builtins.print = _p

        st.write(result["answer"])

        if result.get("tools_used"):
            tools_str = " · ".join(
                f"{TOOL_ICONS.get(t,'🔧')} {t}"
                for t in result["tools_used"]
            )
            st.caption(f"Tools used: {tools_str}")

    for t in result.get("tools_used", []):
        if t in st.session_state.tools_stats:
            st.session_state.tools_stats[t] += 1

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    result["answer"],
        "tools_used": result.get("tools_used", [])
    })
    st.rerun()