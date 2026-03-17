from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from ddgs import DDGS

load_dotenv()

# ── Setup ChromaDB ────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════
# Outil 1 : search_document
# ══════════════════════════════════════════════════════════
@tool
def search_document(query: str) -> str:
    """
    Recherche des informations dans la base de données vectorielle locale (ChromaDB).
    Utilise cet outil pour les questions sur l'indexation, EXPLAIN, les types d'index,
    les performances SQL, et tout sujet lié au document de cours.
    """
    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        return "Aucun résultat trouvé dans le document."

    results = []
    for doc in docs:
        source = f"[{doc.metadata['source']} — page {doc.metadata['page']}]"
        results.append(f"{source}\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


# ══════════════════════════════════════════════════════════
# Outil 2 : search_web
# ══════════════════════════════════════════════════════════
@tool
def search_web(query: str) -> str:
    """
    Recherche des informations sur internet via DuckDuckGo.
    Utilise cet outil pour les questions générales, les actualités,
    ou les sujets non couverts par le document local.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3, region="wt-wt"))

        if not results:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, region="us-en"))

        if not results:
            return "Aucun résultat trouvé sur le web."

        formatted = []
        for r in results:
            formatted.append(f"[{r['title']}]\n{r['body']}\nSource: {r['href']}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Erreur lors de la recherche web : {str(e)}"


# ══════════════════════════════════════════════════════════
# Outil 3 : calculate
# ══════════════════════════════════════════════════════════
@tool
def calculate(expression: str) -> str:
    """
    Effectue des calculs mathématiques simples.
    Utilise cet outil pour les calculs de performance, pourcentages,
    comparaisons numériques. Exemple: '150 / 5' ou '(5000 - 50) / 5000 * 100'
    """
    try:
        # Sécuriser l'eval — autoriser seulement les opérations mathématiques
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "Expression invalide — utilisez uniquement des chiffres et opérateurs (+, -, *, /)."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"


# ══════════════════════════════════════════════════════════
# Outil 4 : summarize
# ══════════════════════════════════════════════════════════
@tool
def summarize(text: str) -> str:
    """
    Résume un texte long en points clés.
    Utilise cet outil quand tu as récupéré beaucoup d'informations
    et que tu veux les condenser avant de répondre.
    """
    # Limiter à 2000 caractères pour éviter de dépasser les tokens
    if len(text) > 2000:
        text = text[:2000] + "..."
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    summary = '\n'.join(f"• {l}" for l in lines[:10])
    return f"Résumé :\n{summary}"


# Liste de tous les outils disponibles
TOOLS = [search_document, search_web, calculate, summarize]