import streamlit as st
from agentic.graph_combined import build_combined_graph
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import builtins
import time

load_dotenv()

st.set_page_config(
    page_title="QueryMind",
    page_icon="⚡",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────
if "messages"           not in st.session_state: st.session_state.messages           = []
if "chat_history"       not in st.session_state: st.session_state.chat_history       = []
if "total_queries"      not in st.session_state: st.session_state.total_queries      = 0
if "graph"              not in st.session_state: st.session_state.graph              = build_combined_graph()
if "saved_convs"        not in st.session_state: st.session_state.saved_convs        = {}   # {id: {title, messages, chat_history, total_queries, tools_stats}}
if "active_conv_id"     not in st.session_state: st.session_state.active_conv_id     = None
if "tools_stats"        not in st.session_state:
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

def save_current_conversation():
    """Persist current session into saved_convs under a stable ID."""
    if not st.session_state.messages:
        return
    conv_id = st.session_state.active_conv_id
    if conv_id is None:
        conv_id = str(time.time_ns())
        st.session_state.active_conv_id = conv_id

    # Title = first user message, truncated
    first_user = next(
        (m["content"] for m in st.session_state.messages if m["role"] == "user"),
        "Conversation"
    )
    title = first_user[:45] + ("…" if len(first_user) > 45 else "")

    st.session_state.saved_convs[conv_id] = {
        "title":         title,
        "messages":      list(st.session_state.messages),
        "chat_history":  list(st.session_state.chat_history),
        "total_queries": st.session_state.total_queries,
        "tools_stats":   dict(st.session_state.tools_stats),
    }

def load_conversation(conv_id):
    """Restore a saved conversation into the active session."""
    conv = st.session_state.saved_convs[conv_id]
    st.session_state.messages       = list(conv["messages"])
    st.session_state.chat_history   = list(conv["chat_history"])
    st.session_state.total_queries  = conv["total_queries"]
    st.session_state.tools_stats    = dict(conv["tools_stats"])
    st.session_state.active_conv_id = conv_id

def start_new_conversation():
    """Save current (if any) and reset to a blank session."""
    save_current_conversation()
    st.session_state.messages       = []
    st.session_state.chat_history   = []
    st.session_state.total_queries  = 0
    st.session_state.active_conv_id = None
    st.session_state.tools_stats    = {k: 0 for k in st.session_state.tools_stats}

## ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    if st.button("✚  Nouvelle conversation", use_container_width=True):
        start_new_conversation()
        st.rerun()

    st.text_input("", placeholder="🔍  Rechercher", label_visibility="collapsed")
    st.divider()

    # Stats
    st.markdown("### 📊 Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Queries",  st.session_state.total_queries)
    c2.metric("Memory",   len(st.session_state.chat_history) // 2)
    c3.metric("Messages", len(st.session_state.messages))
    st.divider()

    # ── Saved conversations ──────────────────────────────
    st.markdown("### 💬 Conversations")
    if st.session_state.saved_convs:
        # Show most recent first
        for conv_id in reversed(list(st.session_state.saved_convs)):
            conv  = st.session_state.saved_convs[conv_id]
            label = conv["title"]
            is_active = (conv_id == st.session_state.active_conv_id)

            col_btn, col_del = st.columns([5, 1])
            btn_type = "primary" if is_active else "secondary"
            if col_btn.button(f"{'▶ ' if is_active else ''}{label}",
                              key=f"conv_{conv_id}",
                              use_container_width=True,
                              type=btn_type):
                save_current_conversation()   # persist current before switching
                load_conversation(conv_id)
                st.rerun()
            if col_del.button("🗑", key=f"del_{conv_id}"):
                del st.session_state.saved_convs[conv_id]
                if st.session_state.active_conv_id == conv_id:
                    start_new_conversation()
                st.rerun()
    else:
        st.caption("Aucune conversation sauvegardée.")
    st.divider()

    # Agent flow
    st.markdown("### 🔀 Agent Flow")
    st.markdown("""
    1. 🧠 **Analyze** — classifie la question
    2. 🤖 **Agent** — choisit les outils
    3. ✅ **Validate** — vérifie la qualité
    4. 🔄 **Retry** — réessaie si besoin
    """)
    st.divider()

    # Knowledge base
    st.markdown("### 📁 Knowledge Base")
    st.info("📄 **document.pdf**\n\n6 pages · 13 chunks · ChromaDB")
    st.divider()

    st.markdown("---")
    st.caption("created by **rahouba**")

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
st.markdown("#  QueryMind")
st.divider()

# Welcome screen
if not st.session_state.messages:
    st.markdown("#### 👋 How can I help you today?")
    st.caption("Ask me anything — documents, web search, summaries, or just a quick question.")

# ── Render full message history ──────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("tools_used"):
                tools_str = "  ·  ".join(
                    f"{TOOL_ICONS.get(t, '🔧')} `{t}`"
                    for t in msg["tools_used"]
                )
                st.caption(f"🔧 Tools: {tools_str}")

            meta_cols = st.columns(3)
            if msg.get("quality_score"):
                stars = "⭐" * msg["quality_score"]
                meta_cols[0].caption(f"Quality: {stars} {msg['quality_score']}/5")
            if msg.get("retry_count") is not None:
                meta_cols[1].caption(f"🔄 Attempts: {msg['retry_count'] + 1}")
            if msg.get("source_type"):
                meta_cols[2].caption(f"🏷️ Type: {msg['source_type']}")

# ════════════════════════════════════════════════════════
# CHAT INPUT
# ════════════════════════════════════════════════════════
question = st.chat_input("Message QueryMind...")

if question:
    st.session_state.total_queries += 1

    # ✅ FIX Bug 1 — save user message FIRST so it appears in history
    st.session_state.messages.append({
        "role":    "user",
        "content": question,
    })

    # Run agent
    with st.spinner("🧠 Thinking..."):
        _p = builtins.print
        builtins.print = lambda *a, **k: _p(*a, **k)
        try:
            result = st.session_state.graph.invoke({
                "question":      question,
                "chat_history":  st.session_state.chat_history,
                "answer":        "",
                "tools_used":    [],
                "quality_score": 0,
                "retry_count":   0,
                "source_type":   ""
            })
        except Exception as e:
            builtins.print = _p
            err = str(e)
            # Rollback — retirer la question déjà sauvegardée
            st.session_state.messages.pop()
            st.session_state.total_queries -= 1
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                st.warning("⚠️ Quota API atteint (20 req/jour). Réessaie demain ou change de clé API.")
            else:
                st.error(f"❌ Erreur : {err}")
            st.stop()
        finally:
            builtins.print = _p

    # Save assistant message
    st.session_state.messages.append({
        "role":          "assistant",
        "content":       result["answer"],
        "tools_used":    result.get("tools_used",    []),
        "quality_score": result.get("quality_score", 0),
        "retry_count":   result.get("retry_count",   0),
        "source_type":   result.get("source_type",   ""),
    })

    # Update stats
    for t in result.get("tools_used", []):
        if t in st.session_state.tools_stats:
            st.session_state.tools_stats[t] += 1

    # Update LangChain memory
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))

    # Limit memory to 10 exchanges
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

    save_current_conversation()

    st.rerun()