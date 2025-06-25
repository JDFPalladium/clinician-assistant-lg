from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")
os.environ.get("LANGSMITH_API_KEY")

from chatlib.state_types import AppState
from chatlib.guidlines_rag_agent_li import rag_retrieve
from chatlib.patient_sql_agent import sql_chain

# from langchain_ollama.chat_models import ChatOllama
# llm = ChatOllama(model="mistral:latest", temperature=0)

tools = [rag_retrieve, sql_chain]
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")
llm_with_tools = llm.bind_tools([rag_retrieve, sql_chain])

# System message
sys_msg = SystemMessage(content="""
                        You are a helpful assistant tasked with helping clinicians
                        meeting with patients. You have two tools available, 
                        rag_retrieve to access information from HIV clinical guidelines,
                        and sql_chain to access patient data.

                        You must respond only with a JSON object specifying the tool to call and its arguments.
                        Do not generate any SQL queries, results or answers yourself. Only the sql_chain
                        tool should do that.
                        When calling a tool, provide only the necessary fields required for that tool to run.
                        Do not include the full state or raw query results in the tool call arguments.
                        For example, include the question and pk_hash, but exclude the query or result.

                        """
                        )

# Assistant Node
def assistant(state: AppState) -> AppState:
    conv = state["conversation"]
    pk_hash = conv.get("pk_hash", None)

    if pk_hash:
        pk_msg = SystemMessage(content=f"The patient identifier (pk_hash) is: {pk_hash}")
        messages = [sys_msg, pk_msg] + state["messages"]
    else:
        messages = [sys_msg] + state["messages"]

    # Get the LLM/tool response
    new_message = llm_with_tools.invoke(messages)
    # Extract the question from the latest HumanMessage, if present
  
    latest_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_question = msg.content
            break

    state['messages'] = state['messages'] + [new_message]
    conv['question'] = latest_question
    state['conversation'] = conv
    return state
    # return {**state, "messages": state['messages'] + [new_message], "question": latest_question}

# Graph
builder = StateGraph(AppState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)

# Specify a thread
memory.delete_thread("25")
config = {"configurable": {"thread_id": "25", "user_id": "1"}}

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

input_state: AppState = {
    "messages": [HumanMessage(content="when was this patient last seen?")],
    "conversation": {
        "question": "",
        "answer": "",
        "pk_hash": "962885FEADB7CCF19A2CC506D39818EC448D5396C4D1AEFDC59873090C7FBF73",
    },
    "query_data": {
        "query": "",
        "result": None,
    },
}

# messages = [HumanMessage(content="how many appointments has this patient had?")]
message_output = react_graph.invoke(input_state, config)

for m in message_output['messages']:
    m.pretty_print()