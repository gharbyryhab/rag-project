from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from agentic.state import AgentState
from agentic.tools import TOOLS

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

llm_with_tools = llm.bind_tools(TOOLS)
tools_map = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """Tu es un assistant expert en bases de données.
Tu as accès à ces outils :
- search_document : chercher dans le cours
- search_web : chercher sur internet
- calculate : faire des calculs
- summarize : résumer un texte

Choisis toujours le bon outil selon la question.
Tiens compte de l'historique de la conversation.
Réponds en français."""


# ══════════════════════════════════════════
# Nœud 1 : analyze
# ══════════════════════════════════════════
def analyze(state: AgentState) -> AgentState:
    print("\n🧠 [analyze] Analyse de la question...")

    prompt = PromptTemplate.from_template("""
Classifie cette question en une seule catégorie :
- "document" : question sur le cours/indexation SQL
- "web"      : question générale sur internet
- "calcul"   : question mathématique
- "résumé"   : demande de résumé

Réponds avec UN seul mot parmi : document, web, calcul, résumé

Question : {question}
Catégorie :""")

    chain    = prompt | llm | StrOutputParser()
    category = chain.invoke({"question": state["question"]}).strip().lower()
    print(f"   → Catégorie : {category}")

    return {**state, "source_type": category}


# ══════════════════════════════════════════
# Nœud 2 : agent (Multi-Tool)
# ══════════════════════════════════════════
def agent(state: AgentState) -> AgentState:
    print("\n🤖 [agent] Sélection et exécution des outils...")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.get("chat_history", []),
        HumanMessage(content=state["question"])
    ]

    tools_used = []
    max_iter   = 5
    iteration  = 0

    while iteration < max_iter:
        iteration += 1
        response  = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            content = response.content
            if isinstance(content, list):
                content = "\n".join(
                    b["text"] for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            print(f"   → {len(tools_used)} outil(s) utilisé(s)")
            return {**state, "answer": content, "tools_used": tools_used}

        messages.append(response)

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            tools_used.append(name)

            preview = str(list(args.values())[0])[:40] if args else ""
            print(f"   🔧 {name}({preview}...)")

            fn     = tools_map.get(name)
            result = fn.invoke(args) if fn else f"Outil {name} introuvable"

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

    return {**state, "answer": "Je n'ai pas pu répondre.", "tools_used": tools_used}


# ══════════════════════════════════════════
# Nœud 3 : validate
# ══════════════════════════════════════════
def validate(state: AgentState) -> AgentState:
    print("\n✅ [validate] Vérification de la qualité...")

    prompt = PromptTemplate.from_template("""
Évalue cette réponse sur une échelle de 1 à 5.
Réponds UNIQUEMENT avec un chiffre entre 1 et 5.

Question : {question}
Réponse  : {answer}

Score (1-5) :""")

    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({
    "question": state["question"],
    "answer":   state.get("answer", "")
})

    try:
        score = int(result.strip()[0])
    except:
        score = 3

    print(f"   → Score : {score}/5")
    return {**state, "quality_score": score}


# ══════════════════════════════════════════
# Nœud 4 : retry
# ══════════════════════════════════════════
def retry(state: AgentState) -> AgentState:
    print("\n🔄 [retry] Reformulation...")

    prompt = PromptTemplate.from_template("""
La réponse précédente était insuffisante.
Reformule la question pour obtenir une meilleure réponse.
Réponds uniquement avec la question reformulée, sans explication.

Question originale : {question}
Question reformulée :""")

    chain        = prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]}).strip()

    print(f"   → Nouvelle question : {new_question}")
    return {
        **state,
        "question":    new_question,
        "retry_count": state.get("retry_count", 0) + 1
    }