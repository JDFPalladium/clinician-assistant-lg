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

import subprocess
import socket

def is_ollama_running(host="127.0.0.1", port=11434):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0

if not is_ollama_running():
    # Start ollama serve as a background process
    subprocess.Popen(["ollama", "serve"])

load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")
os.environ.get("LANGSMITH_API_KEY")

from chatlib.state_types import AppState
from chatlib.guidlines_rag_agent_li import rag_retrieve
from chatlib.patient_sql_agent import sql_chain

tools = [rag_retrieve, sql_chain]
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")
llm_with_tools = llm.bind_tools([rag_retrieve, sql_chain])

# System message
sys_msg = SystemMessage(content="""
                        You are a helpful assistant tasked with helping clinicians
                        meeting with patients. You have two tools available, 
                        rag_retrieve to access information from HIV clinical guidelines,
                        and sql_chain to access patient data. 

                        In some cases, you may need to use both tools to answer a question.
                        If you need to use both tools, first call rag_retrieve to get the relevant information,
                        then call sql_chain to get the patient data, and finally combine the results
                        to provide a complete answer. For example, if the question is about whether 
                        a patient is on the correct treatment, you might first retrieve the treatment guidelines
                        using rag_retrieve, then check the patient's treatment history using sql_chain.

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

def chat_with_patient(question: str, pk_hash: str, thread_id: str = None):
    # Generate or reuse thread_id for session persistence
    if thread_id is None or thread_id == "":
        thread_id = str(uuid.uuid4())

    # Prepare input state with new user message and pk_hash
    input_state: AppState = {
        "messages": [HumanMessage(content=question)],
        "conversation": {
            "question": "",
            "answer": "",
            "pk_hash": pk_hash if pk_hash else None,
        },
        "query_data": {
            "query": "",
            "result": None,
        }
    }

    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}

    # Invoke the graph with persistent state
    output_state = react_graph.invoke(input_state, config)

    # Extract assistant reply from messages
    assistant_message = output_state["messages"][-1].content

    return assistant_message, thread_id

with gr.Blocks() as demo:
    question_input = gr.Textbox(label="Question")
    pk_hash_input = gr.Textbox(label="Patient pk_hash")
    thread_id_state = gr.State()  # to store thread_id between calls
    output_chat = gr.Textbox(label="Assistant Response")

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        chat_with_patient,
        inputs=[question_input, pk_hash_input, thread_id_state],
        outputs=[output_chat, thread_id_state],
    )

demo.launch()
