from typing_extensions import TypedDict, Annotated, NotRequired
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    rag_result: str
    answer: str
    last_answer: Optional[str] = None
    last_user_message: Optional[str] = None
    last_tool: Optional[str] = None
    idsr_disclaimer_shown: bool = False
    summary: Optional[str] = None
