from langgraph.graph import StateGraph, END
from agentic.state import AgentState
from agentic.nodes_combined import analyze, agent, validate, retry


def decide_after_validate(state: AgentState) -> str:
    score       = state.get("quality_score", 3)
    retry_count = state.get("retry_count", 0)

    if score >= 3:
        print(f"   → ✅ Qualité ok ({score}/5)")
        return "end"
    elif retry_count < 2:
        print(f"   → ⚠️ Qualité insuffisante ({score}/5) — retry")
        return "retry"
    else:
        print(f"   → 🛑 Trop d'essais — fin forcée")
        return "end"


def build_combined_graph():
    graph = StateGraph(AgentState)

    # Nœuds
    graph.add_node("analyze",  analyze)
    graph.add_node("agent",    agent)
    graph.add_node("validate", validate)
    graph.add_node("retry",    retry)

    # Point d'entrée
    graph.set_entry_point("analyze")

    # Transitions fixes
    graph.add_edge("analyze",  "agent")
    graph.add_edge("agent",    "validate")
    graph.add_edge("retry",    "agent")

    # Transition conditionnelle
    graph.add_conditional_edges(
        "validate",
        decide_after_validate,
        {
            "end":   END,
            "retry": "retry",
        }
    )

    return graph.compile()