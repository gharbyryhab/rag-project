from agentic.graph import build_graph

graph = build_graph()

print("=" * 55)
print("   🤖 Agentic RAG — Bases de Données")
print("   Tape 'exit' pour quitter")
print("=" * 55)

while True:
    print()
    question = input("❓ Ta question : ").strip()

    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        print("👋 Au revoir !")
        break

    result = graph.invoke({
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "source_type": ""
    })

    print("\n" + "=" * 55)
    print("✅ Réponse finale :")
    print("=" * 55)
    print(result["generation"])

    if result["documents"]:
        print("\n📚 Sources utilisées :")
        sources_vues = set()
        for doc in result["documents"]:
            cle = (doc.metadata['source'], doc.metadata['page'])
            if cle not in sources_vues:
                if doc.metadata['page'] == "web":
                    print(f"   🌐 {doc.metadata['source']}")
                else:
                    print(f"   • {doc.metadata['source']} — page {doc.metadata['page']}")
                sources_vues.add(cle)

    print("-" * 55)