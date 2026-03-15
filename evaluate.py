from dotenv import load_dotenv
import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Setup RAG ─────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
    transport="rest"
)

vectorstore = Chroma(
    persist_directory="vectorstore/",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# ── Chain RAG ─────────────────────────────────────────────
prompt_rag = PromptTemplate.from_template("""
Tu es un assistant pédagogique. Réponds à la question en te basant
uniquement sur le contexte fourni. Si la réponse n'est pas dans
le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

# ── Prompt d'évaluation ───────────────────────────────────
prompt_eval = PromptTemplate.from_template("""
Tu es un évaluateur de système RAG. Évalue la réponse suivante.

Question : {question}
Contexte récupéré : {context}
Réponse générée : {reponse}

Évalue sur 3 critères, chacun noté de 1 à 5 :
1. Pertinence du contexte : le contexte récupéré est-il lié à la question ?
2. Fidélité : la réponse est-elle basée uniquement sur le contexte ?
3. Complétude : la réponse couvre-t-elle bien tous les aspects de la question ?

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après, dans ce format exact :
{{
  "pertinence_contexte": <note 1-5>,
  "fidelite": <note 1-5>,
  "completude": <note 1-5>,
  "commentaire": "<une phrase d'explication>"
}}""")

eval_chain = prompt_eval | llm | StrOutputParser()

# ── Jeu de test ───────────────────────────────────────────
questions_test = [
    "Quels sont les types d'index disponibles ?",
    "Qu'est-ce que la commande EXPLAIN ?",
    "Quel est le gain de performance avec un index B-Tree ?",
    "Quelles sont les bonnes pratiques d'indexation ?",
]

# ── Évaluation ────────────────────────────────────────────
print("=" * 60)
print("   📊 Évaluation du RAG")
print("=" * 60)

resultats = []
scores_totaux = {"pertinence_contexte": 0, "fidelite": 0, "completude": 0}

for i, question in enumerate(questions_test):
    print(f"\n🔍 Test {i+1}/{len(questions_test)} : {question}")

    # Récupérer contexte et réponse
    docs = retriever.invoke(question)
    contexte = format_docs(docs)
    reponse = rag_chain.invoke(question)

    # Évaluer
    eval_brut = eval_chain.invoke({
        "question": question,
        "context": contexte,
        "reponse": reponse
    })

    # Parser le JSON
    try:
        # Nettoyer les éventuels backticks
        eval_brut = eval_brut.replace("```json", "").replace("```", "").strip()
        eval_json = json.loads(eval_brut)

        print(f"   📝 Pertinence contexte : {eval_json['pertinence_contexte']}/5")
        print(f"   ✅ Fidélité            : {eval_json['fidelite']}/5")
        print(f"   📋 Complétude          : {eval_json['completude']}/5")
        print(f"   💬 {eval_json['commentaire']}")

        for cle in scores_totaux:
            scores_totaux[cle] += eval_json[cle]

        resultats.append({"question": question, "scores": eval_json})

    except Exception as e:
        print(f"   ⚠️ Erreur de parsing : {e}")
        print(f"   Réponse brute : {eval_brut}")

# ── Résumé final ──────────────────────────────────────────
n = len(resultats)
if n > 0:
    print("\n" + "=" * 60)
    print("   📈 Scores moyens")
    print("=" * 60)
    for cle, total in scores_totaux.items():
        moyenne = total / n
        print(f"   {cle:25} : {moyenne:.1f}/5")
    
    score_global = sum(scores_totaux.values()) / (n * 3)
    print(f"\n   🏆 Score global : {score_global:.1f}/5")
    print("=" * 60)