from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    generation: str
    retry_count: int
    source_type: str
    chat_history: List[BaseMessage]  # ← nouveau champ pour l'historique du chat