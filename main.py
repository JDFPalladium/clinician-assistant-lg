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

from chatlib.state_types import State
from chatlib.guidlines_rag_agent_li import rag_retrieve
from chatlib.patient_sql_agent import sql_chain

tools = [rag_retrieve, sql_chain]
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")
llm_with_tools = llm.bind_tools([rag_retrieve, sql_chain])

# System message
sys_msg = SystemMessage(content="""
                        You are a helpful assistant tasked with helping clinicians
                        meeting with patients. You have two tools available, 
                        one to access information from HIV clinical guidelines, the other is
                        a SQL tool to access patient data.

                        You must respond only with a JSON object specifying the tool to call and its arguments.
                        Do not generate any SQL queries or answers yourself.
                        """
                        )

# Assistant Node
def assistant(state: State) -> State:
    pk_hash = state.get("pk_hash", None)

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
    return {**state, "messages": state['messages'] + [new_message], "question": latest_question}

# Graph
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)

# Specify a thread
memory.delete_thread("25")
config = {"configurable": {"thread_id": "25", "user_id": "1"}}

# initialize state with patient pk hash
input_state:State = {
    "messages": [HumanMessage(content="how many visits were recorded in 2024?")],
    "question": "",
    "rag_result": "",
    "query": "",
    "result": "",
    "answer": "",
    "pk_hash": "962885FEADB7CCF19A2CC506D39818EC448D5396C4D1AEFDC59873090C7FBF73"
}

# messages = [HumanMessage(content="how many appointments has this patient had?")]
message_output = react_graph.invoke(input_state, config)

for m in message_output['messages']:
    m.pretty_print()