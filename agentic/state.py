from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    generation: str
    retry_count: int
    source_type: str