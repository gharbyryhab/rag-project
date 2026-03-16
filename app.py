import streamlit as st
from agentic.graph import build_graph
from dotenv import load_dotenv

load_dotenv()

# ── Configuration de la page ──────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="centered"
)

# ── Titre ─────────────────────────────────────────────────
st.title("🤖 Agentic RAG — Bases de Données")
st.caption("Posez vos questions sur le document ou sur le web !")

# ── Initialisation du graph et historique ─────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Affichage de l'historique des messages ────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources utilisées"):
                for source in msg["sources"]:
                    st.markdown(source)

# ── Input utilisateur ─────────────────────────────────────
question = st.chat_input("Posez votre question...")

if question:
    # Afficher la question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Lancer le graph
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):

            # Affichage des étapes en temps réel
            status_placeholder = st.empty()

            try:
                # Stream les étapes du graph
                steps_log = []
                for event in st.session_state.graph.stream({
                    "question": question,
                    "rewritten_question": "",
                    "documents": [],
                    "generation": "",
                    "retry_count": 0,
                    "source_type": ""
                }):
                    for node_name, node_output in event.items():
                        if node_name == "retrieve":
                            steps_log.append("📥 Recherche dans ChromaDB...")
                        elif node_name == "grade_documents":
                            pertinent = bool(node_output.get("documents"))
                            steps_log.append(f"🔎 Pertinent : {'✅ oui' if pertinent else '❌ non'}")
                        elif node_name == "rewrite_query":
                            steps_log.append("✏️ Reformulation de la question...")
                        elif node_name == "web_search":
                            steps_log.append("🌐 Recherche sur le web...")
                        elif node_name == "generate":
                            steps_log.append("🤖 Génération de la réponse...")

                        status_placeholder.markdown("\n\n".join(steps_log))

                # Récupérer le résultat final
                result = st.session_state.graph.invoke({
                    "question": question,
                    "rewritten_question": "",
                    "documents": [],
                    "generation": "",
                    "retry_count": 0,
                    "source_type": ""
                })

                status_placeholder.empty()

                # Afficher la réponse
                st.markdown(result["generation"])

                # Préparer les sources
                sources = []
                sources_vues = set()
                for doc in result["documents"]:
                    cle = (doc.metadata['source'], doc.metadata['page'])
                    if cle not in sources_vues:
                        if doc.metadata['page'] == "web":
                            sources.append(f"🌐 [{doc.metadata['source']}]({doc.metadata['source']})")
                        else:
                            sources.append(f"📄 {doc.metadata['source']} — page {doc.metadata['page']}")
                        sources_vues.add(cle)

                # Afficher les sources
                if sources:
                    with st.expander("📚 Sources utilisées"):
                        for source in sources:
                            st.markdown(source)

                # Badge source type
                source_type = result.get("source_type", "")
                if source_type == "web":
                    st.info("🌐 Réponse basée sur une recherche web")
                elif source_type == "document":
                    st.success("📄 Réponse basée sur le document")

                # Sauvegarder dans l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["generation"],
                    "sources": sources
                })

            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Informations")
    st.metric("Messages échangés", len(st.session_state.messages))

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**📁 Document chargé :**")
    st.markdown("• document.pdf")

    st.divider()
    st.markdown("**🔀 Flux de l'agent :**")
    st.markdown("""
    1. 📥 Retrieve (ChromaDB)
    2. 🔎 Grade documents
    3. ✏️ Rewrite query *(si non pertinent)*
    4. 🌐 Web search *(si échec)*
    5. 🤖 Generate
    """)