from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")
os.environ.get("LANGSMITH_API_KEY")

from chatlib.state_types import AppState
from chatlib.guidlines_rag_agent_li import rag_retrieve
from chatlib.patient_all_data import sql_chain
from chatlib.idsr_check import idsr_check

# from langchain_ollama.chat_models import ChatOllama
# llm = ChatOllama(model="mistral:latest", temperature=0)

tools = [rag_retrieve, sql_chain, idsr_check]
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")
llm_with_tools = llm.bind_tools([rag_retrieve, sql_chain, idsr_check])

# System message
sys_msg = SystemMessage(content="""
You are a helpful assistant supporting clinicians during patient visits. You have three tools:

- rag_retrieve: to access HIV clinical guidelines
- sql_chain: to query patient data from the SQL database
- idsr_check: to check if the patient case description matches any known diseases

There are three types of questions you may receive:
1. Questions about patients (e.g., "When should this patient switch regimens?" or "What is their viral load history?")
2. Questions about HIV clinical guidelines (e.g., "What are the latest guidelines for changing ART regimens?")
3. Questions about disease identification based on patient case descriptions (e.g., "Should I be concerned about certain diseases with this patient?")

When a clinician asks about patients, first use rag_retrieve to get relevant guideline context, then use sql_chain to query the patient's data, combining information as needed.

When a clinician asks about guidelines, use rag_retrieve to provide the latest HIV clinical guidelines.

When a clinician asks about disease identification, use idsr_check to match case descriptions against disease definitions.

Respond only with a JSON object specifying the tool to call and its arguments, for example:
{
  "tool": "rag_retrieve",
  "args": {"query": "latest ART regimen guidelines"}
}

Keep responses concise and focused. The clinician is a healthcare professional; do not suggest consulting one.

If the clinician's question is unclear, ask for clarification.

Do not include any text outside the JSON response.
""")


# Assistant Node
def assistant(state: AppState) -> AppState:

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

    state['messages'] = state['messages'] + [new_message]
    state['question'] = latest_question
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
config = {"configurable": {"thread_id": "30"}}

# initialize state with patient pk hash
input_state:AppState = {
    "messages": [HumanMessage(content="summarize the patient's clinical visits")],
    "question": "",
    "rag_result": "",
    "answer": "",
    "pk_hash": "962885FEADB7CCF19A2CC506D39818EC448D5396C4D1AEFDC59873090C7FBF73"
}


# messages = [HumanMessage(content="how many appointments has this patient had?")]
message_output = react_graph.invoke(input_state, config)

for m in message_output['messages']:
    m.pretty_print()