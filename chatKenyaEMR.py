import gradio as gr
import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Initialize your graph and checkpointer once
memory = MemorySaver()
# react_graph = ... your compiled LangGraph graph with memory as checkpointer

def chat_with_patient(question: str, pk_hash: str, thread_id: str = None):
    # Generate or reuse thread_id for session persistence
    if thread_id is None or thread_id == "":
        thread_id = str(uuid.uuid4())

    # Prepare input state with new user message and pk_hash
    input_state = {
        "messages": [HumanMessage(content=question)],
        "question": "",
        "rag_result": "",
        "query": "",
        "result": "",
        "answer": "",
        "pk_hash": pk_hash,
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
