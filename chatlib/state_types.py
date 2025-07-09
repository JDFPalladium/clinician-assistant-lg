from typing_extensions import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# class ConversationState(TypedDict):
#     question: str
#     answer: str
#     rag_result: str
#     pk_hash: Optional[str]

# class QueryState(TypedDict):
#     query: str
#     result: Optional[str]

# class AppState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#     conversation: ConversationState
#     query_data: QueryState

# class SqlChainOutputModel(BaseModel):
#     messages: List[AnyMessage] = Field(...)
#     conversation: ConversationState = Field(...)

class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    rag_result: str
    answer: str
    last_answer: Optional[str] = None
    last_tool: Optional[str] = None
    pk_hash: str
