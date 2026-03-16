from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from ddgs import DDGS
from agentic.state import AgentState

load_dotenv()

# ── Setup partagé ─────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════
# Nœud 1 : retrieve
# Récupère les chunks pertinents depuis ChromaDB
# ══════════════════════════════════════════════════════════
def retrieve(state: AgentState) -> AgentState:
    print("\n📥 [retrieve] Recherche dans ChromaDB...")
    question = state.get("rewritten_question") or state["question"]
    docs = retriever.invoke(question)
    print(f"   → {len(docs)} chunks récupérés")
    return {**state, "documents": docs}


# ══════════════════════════════════════════════════════════
# Nœud 2 : grade_documents
# L'agent évalue si les chunks sont pertinents
# ══════════════════════════════════════════════════════════
def grade_documents(state: AgentState) -> AgentState:
    print("\n🔎 [grade_documents] Évaluation de la pertinence...")

    question = state.get("rewritten_question") or state["question"]
    docs = state["documents"]

    prompt = PromptTemplate.from_template("""
Tu es un évaluateur. La question suivante est-elle couverte par ce contexte ?
Réponds uniquement par 'oui' ou 'non'.

Question : {question}
Contexte : {context}

Réponse (oui/non) :""")

    chain = prompt | llm | StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in docs)
    result = chain.invoke({"question": question, "context": context})
    relevant = "oui" in result.lower()

    print(f"   → Pertinent : {'✅ oui' if relevant else '❌ non'}")
    return {**state, "documents": docs if relevant else [], "source_type": "document" if relevant else ""}


# ══════════════════════════════════════════════════════════
# Nœud 3 : rewrite_query
# Reformule la question si les chunks ne sont pas pertinents
# ══════════════════════════════════════════════════════════
def rewrite_query(state: AgentState) -> AgentState:
    print("\n✏️  [rewrite_query] Reformulation de la question...")

    prompt = PromptTemplate.from_template("""
Tu es un expert en recherche d'information. Reformule cette question
pour améliorer la récupération de documents pertinents.
Réponds uniquement avec la question reformulée, sans explication.

Question originale : {question}

Question reformulée :""")

    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]})

    print(f"   → Nouvelle question : {new_question.strip()}")
    retry_count = state.get("retry_count", 0) + 1
    return {**state, "rewritten_question": new_question.strip(), "retry_count": retry_count}


# ══════════════════════════════════════════════════════════
# Nœud 4 : web_search
# Cherche sur le web si le document ne contient pas la réponse
# ══════════════════════════════════════════════════════════
def web_search(state: AgentState) -> AgentState:
    print("\n🌐 [web_search] Recherche sur le web...")

    question = state.get("rewritten_question") or state["question"]

    with DDGS() as ddgs:
        results = list(ddgs.text(question, max_results=3))

    web_docs = []
    for r in results:
        web_docs.append(Document(
            page_content=r["body"],
            metadata={"source": r["href"], "page": "web"}
        ))
        print(f"   → {r['title'][:60]}...")

    return {**state, "documents": web_docs, "source_type": "web"}


# ══════════════════════════════════════════════════════════
# Nœud 5 : generate
# Génère la réponse finale avec le contexte récupéré
# ══════════════════════════════════════════════════════════
def generate(state: AgentState) -> AgentState:
    print("\n🤖 [generate] Génération de la réponse...")

    prompt = PromptTemplate.from_template("""
Tu es un assistant pédagogique. Réponds à la question en te basant
uniquement sur le contexte fourni. Si la réponse n'est pas dans
le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :""")

    chain = prompt | llm | StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in state["documents"])
    question = state.get("rewritten_question") or state["question"]
    response = chain.invoke({"context": context, "question": question})

    return {**state, "generation": response}