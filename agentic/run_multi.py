from agentic.multi_agent import run_multi_agent
from langchain_core.messages import HumanMessage, AIMessage

print("=" * 55)
print("   🛠️  Multi-Tool Agent — Bases de Données")
print("   Outils : search_document | search_web | calculate | summarize")
print("   Tape 'exit' pour quitter")
print("=" * 55)

chat_history = []

while True:
    print()
    question = input("❓ Ta question : ").strip()

    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        print("👋 Au revoir !")
        break

    result = run_multi_agent(question, chat_history)

    print("\n" + "=" * 55)
    print("✅ Réponse finale :")
    print("=" * 55)
    print(result["answer"])

    if result["tools_used"]:
        print(f"\n🔧 Outils utilisés : {' → '.join(result['tools_used'])}")

    # Mettre à jour l'historique
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))

    print(f"🧠 Historique : {len(chat_history)//2} échange(s)")
    print("-" * 55)