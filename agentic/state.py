from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    question:      str
    chat_history:  List[BaseMessage]
    answer:        str
    tools_used:    List[str]
    quality_score: int
    retry_count:   int
    source_type:   str