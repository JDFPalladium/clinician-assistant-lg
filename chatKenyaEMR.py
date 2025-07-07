import gradio as gr
import uuid
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Initialize your graph and checkpointer once - eventually make this persistent
memory = MemorySaver()

load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")
os.environ.get("LANGSMITH_API_KEY")

from chatlib.state_types import AppState
from chatlib.guidlines_rag_agent_li import rag_retrieve
from chatlib.patient_all_data import sql_chain
from chatlib.idsr_check import idsr_check

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

def chat_with_patient(question: str, pk_hash: str, thread_id: str = None):
    # Generate or reuse thread_id for session persistence
    if thread_id is None or thread_id == "":
        thread_id = str(uuid.uuid4())

    # Prepare input state with new user message and pk_hash
    # initialize state with patient pk hash
    input_state:AppState = {
        "messages": [HumanMessage(content=question)],
        "question": "",
        "rag_result": "",
        "answer": "",
        "pk_hash": pk_hash
    }

    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}

    # Invoke the graph with persistent state
    output_state = react_graph.invoke(input_state, config)

    for m in output_state['messages']:
        m.pretty_print()

    # Extract assistant reply from messages
    assistant_message = output_state["messages"][-1].content

    # extract message history for the thread
    thread_messages = []
    for msg in output_state['messages']:
        if isinstance(msg, HumanMessage) or isinstance(msg, SystemMessage):
            thread_messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "system",
                "content": msg.content
            })

    return assistant_message, thread_messages,  thread_id

with gr.Blocks() as demo:
    question_input = gr.Textbox(label="Question")
    pk_hash_input = gr.Textbox(label="Patient pk_hash")
    thread_id_state = gr.State()  # to store thread_id between calls
    output_chat = gr.Textbox(label="Assistant Response")
    output_message_history = gr.Textbox(label="Message History", max_lines=10)

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        chat_with_patient,
        inputs=[question_input, pk_hash_input, thread_id_state],
        outputs=[output_chat, output_message_history, thread_id_state],
    )

demo.launch()
