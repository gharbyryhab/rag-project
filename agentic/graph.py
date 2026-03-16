from langgraph.graph import StateGraph, END
from agentic.state import AgentState
from agentic.nodes import retrieve, grade_documents, rewrite_query, generate, web_search


def decide_after_grading(state: AgentState) -> str:
    """
    Décision après l'évaluation des chunks :
    - Si pertinents → générer
    - Si non pertinents et 0 essai → reformuler
    - Si non pertinents et 1+ essai → web_search
    """
    if state["documents"]:
        return "generate"
    elif state.get("retry_count", 0) < 1:
        return "rewrite_query"
    else:
        return "web_search"


def build_graph():
    graph = StateGraph(AgentState)

    # ── Ajouter les nœuds ─────────────────────────────────
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    # ── Point d'entrée ────────────────────────────────────
    graph.set_entry_point("retrieve")

    # ── Transitions ───────────────────────────────────────
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        }
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()