# type: ignore
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Optional


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    pk_hash: str
    sitecode: str
    rag_result: str
    rag_sources: Optional[str]  # Added to store retrieved sources
    answer: str
    last_answer: Optional[str] = None
    last_user_message: Optional[str] = None
    last_tool: Optional[str] = None
    idsr_disclaimer_shown: bool = False
    summary: Optional[str] = None
    context: Optional[str] = None
    context_versions: dict[str, int] = {}
    last_context_injected_versions: dict[str, int] = {}
    context_version_ready_for_injection: int = 0
    context_first_response_sent: bool = True
