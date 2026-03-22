from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from agentic.tools import TOOLS

load_dotenv()

# ── LLM avec outils bindés ────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

llm_with_tools = llm.bind_tools(TOOLS)

# Map nom → fonction
tools_map = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """Tu es un assistant expert en bases de données.
Tu as accès à ces outils :
- search_document : pour chercher dans le cours sur l'indexation SQL
- search_web : pour chercher sur internet
- calculate : pour faire des calculs mathématiques
- summarize : pour résumer un texte long

Utilise toujours search_document en premier pour les questions sur l'indexation.
Si le document ne contient pas la réponse, utilise search_web.
Réponds toujours en français."""


def parse_content(content) -> str:
    """Parse le contenu de la réponse Gemini quel que soit son format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(content)


def run_multi_agent(question: str, chat_history: list = []) -> dict:
    print(f"\n🤖 [multi_agent] Analyse de la question...")

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + chat_history + [
        HumanMessage(content=question)
    ]

    tools_used = []
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        response = llm_with_tools.invoke(messages)

        # ── Réponse finale (pas d'appel d'outil) ─────────
        if not response.tool_calls:
            print(f"   ✅ Réponse finale générée")
            return {
                "answer": parse_content(response.content),
                "tools_used": tools_used
            }

        # ── Traiter les appels d'outils ───────────────────
        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Afficher l'outil appelé
            first_arg = list(tool_args.values())[0] if tool_args else ""
            preview = str(first_arg)[:50]
            print(f"   🔧 Outil appelé : {tool_name}({preview}...)")
            tools_used.append(tool_name)

            # Exécuter l'outil
            tool_fn = tools_map.get(tool_name)
            if tool_fn:
                try:
                    tool_result = tool_fn.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Erreur lors de l'exécution de {tool_name} : {str(e)}"
            else:
                tool_result = f"Outil '{tool_name}' non trouvé."

            # Ajouter le résultat au contexte
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

    # ── Trop d'itérations ─────────────────────────────────
    return {
        "answer": "Désolé, je n'ai pas pu trouver une réponse après plusieurs tentatives.",
        "tools_used": tools_used
    }