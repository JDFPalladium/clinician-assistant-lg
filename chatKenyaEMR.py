import gradio as gr
import uuid
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
from chatlib.phi_filter import detect_and_redact_phi
from chatlib.assistant_node import assistant

tools = [rag_retrieve, sql_chain, idsr_check]
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")
llm_with_tools = llm.bind_tools([rag_retrieve, sql_chain, idsr_check])

# System message
sys_msg = SystemMessage(content="""
You are a helpful assistant supporting clinicians during patient visits. You have three tools:

- rag_retrieve: to access HIV clinical guidelines
- sql_chain: to query patient data from the SQL database
- idsr_check: to check if the patient case description matches any known diseases

When calling a tool, respond only with a JSON object specifying the tool to call and its minimal arguments, for example:
{
  "tool": "idsr_check",
  "args": {"query": "patient vaginal bleeding"}
}

Do not pass the entire state as an argument.

Keep responses concise and focused. The clinician is a healthcare professional; do not suggest consulting one.

If the clinician's question is unclear, ask for clarification.

Do not include any text outside the JSON response.
""")

# Graph
builder = StateGraph(AppState)
builder.add_node(
    "assistant",
    lambda state: assistant(state, sys_msg, llm, llm_with_tools)
)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)

def chat_with_patient(question: str, thread_id: str = None):
    # Generate or reuse thread_id for session persistence
    if thread_id is None or thread_id == "":
        thread_id = str(uuid.uuid4())

    # Check input for PHI and redact if necessary
    question = detect_and_redact_phi(question)["redacted_text"]
    print(question)
    # Prepare input state with new user message and pk_hash
    # initialize state with patient pk hash
    input_state:AppState = {
        "messages": [HumanMessage(content=question)],
        "question": "",
        "rag_result": "",
        "answer": ""
    }

    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}

    # Invoke the graph with persistent state
    output_state = react_graph.invoke(input_state, config)

    for m in output_state['messages']:
        m.pretty_print()

    # Extract the last AImessage 
    assistant_message = output_state["messages"][-1].content

    return assistant_message, thread_id

with gr.Blocks() as demo:
    question_input = gr.Textbox(label="Question")
    thread_id_state = gr.State()  # to store thread_id between calls
    output_chat = gr.Textbox(label="Assistant Response")

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        chat_with_patient,
        inputs=[question_input, thread_id_state],
        outputs=[output_chat, thread_id_state],
    )

demo.launch()
