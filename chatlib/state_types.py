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
    query: str
    result: str
    answer: str
    pk_hash: str

# initialize state with patient pk hash
# input_state:State = {
#     "messages": [HumanMessage(content="was this person typically late or on time to their visits?")],
#     "question": "",
#     "rag_result": "",
#     "query": "",
#     "result": "",
#     "answer": "",
#     "pk_hash": "962885FEADB7CCF19A2CC506D39818EC448D5396C4D1AEFDC59873090C7FBF73"
# }