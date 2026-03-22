from agentic.graph_combined import build_combined_graph
from langchain_core.messages import HumanMessage, AIMessage

graph = build_combined_graph()

print("=" * 55)
print("   🚀 Combined Agent — LangGraph + Multi-Tool")
print("   🧠 Mémoire conversationnelle activée")
print("   Tape 'exit' pour quitter")
print("=" * 55)

# Mémoire persistante
chat_history = []

while True:
    print()
    question = input("❓ Ta question : ").strip()

    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        print("👋 Au revoir !")
        break

    result = graph.invoke({
        "question":      question,
        "chat_history":  chat_history,
        "answer":        "",
        "tools_used":    [],
        "quality_score": 0,
        "retry_count":   0,
        "source_type":   ""
    })

    print("\n" + "=" * 55)
    print("✅ Réponse finale :")
    print("=" * 55)
    print(result["answer"])

    if result["tools_used"]:
        print(f"\n🔧 Outils    : {' → '.join(result['tools_used'])}")

    print(f"⭐ Qualité   : {result['quality_score']}/5")
    print(f"🔄 Essais    : {result['retry_count'] + 1}")
    print(f"🧠 Mémoire   : {len(chat_history)//2 + 1} échange(s)")
    print("-" * 55)

    # Mettre à jour la mémoire
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))

    # Garder max 10 échanges (20 messages)
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]