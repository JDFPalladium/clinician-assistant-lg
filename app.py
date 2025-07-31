# type: ignore
import gradio as gr
import uuid
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver


memory = MemorySaver()

if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.0, model="gpt-4o")

from chatlib.state_types import AppState
from chatlib.guidlines_rag_agent_li import rag_retrieve
from chatlib.patient_all_data import sql_chain
from chatlib.idsr_check import idsr_check
from chatlib.idsr_definition import idsr_define
from chatlib.phi_filter import detect_and_redact_phi
from chatlib.assistant_node import assistant


def rag_retrieve_tool(query):
    """Retrieve relevant HIV clinical guidelines for the given query."""
    result = rag_retrieve(query, llm=llm)
    return {
        "rag_result": result.get("rag_result", ""),
        "rag_sources": result.get("rag_sources", []),
        "last_tool": "rag_retrieve",
    }


def sql_chain_tool(query, rag_result, pk_hash):
    """Query patient data from the SQL database and summarize results."""
    result = sql_chain(query, llm=llm, rag_result=rag_result, pk_hash=pk_hash)
    return {"answer": result.get("answer", ""), "last_tool": "sql_chain"}


def idsr_check_tool(query, sitecode):
    """Check if the patient case description matches any known diseases."""
    result = idsr_check(query, llm=llm, sitecode=sitecode)

    return {
        "answer": result.get("answer", ""),
        "last_tool": "idsr_check",
        "context": result.get("context", None),
    }

def idsr_define_tool(query):
    """Retrieve disease definition based on the query."""
    result = idsr_define(query, llm=llm)
    return {
        "answer": result.get("answer", ""),
        "last_tool": "idsr_define"
    }

tools = [rag_retrieve_tool, sql_chain_tool, idsr_check_tool, idsr_define_tool]
llm_with_tools = llm.bind_tools(tools)


sys_msg = SystemMessage(
    content="""
You are a helpful assistant supporting clinicians during patient visits. When a patient ID is provided, the clinician is meeting with that HIV-positive patient and may inquire about their history, lab results, or medications. If no patient ID is provided, the clinician may be asking general HIV clinical questions or presenting symptoms for a new patient.

You have access to four tools to help you answer the clinician's questions. 

- rag_retrieve_tool: to access HIV clinical guidelines
- sql_chain_tool: to access HIV data about the patient with whom the clinician is meeting. For straightforward factual questions about the patient, you may call sql_chain directly. For questions requiring clinical interpretation or classification, first call rag_retrieve to get relevant clinical guideline context, then include that context when calling sql_chain.
- idsr_check_tool: to check if the patient case description matches any known diseases.
- idsr_define_tool: to retrieve the official case definition of a disease when the clinician asks about it (e.g., “What is the description of cholera?”). Do not use this tool for analyzing symptom descriptions — use `idsr_check_tool` for that.

When a tool is needed, respond only with a JSON object specifying the tool to call and its minimal arguments, for example:
{
  "tool": "rag_retrieve_tool",
  "args": {
    "query": "patient vaginal bleeding",
  }
}

When calling the "sql_chain" tool, always include the following arguments in the JSON response:

- "query": the clinician's question
- "rag_result": the clinical guideline context obtained from rag_retrieve
- "pk_hash": the patient identifier string

For example:

{
  "tool": "sql_chain_tool",
  "args": {
    "query": "What is the patient's latest lab results?",
    "rag_result": "<clinical guideline context>",
    "pk_hash": "patient123"
  }
}

When calling the "idsr_check_tool" tool, always include the following arguments in the JSON response:

- "query": the clinician's question
- "sitecode": the site code string

For example:

{
  "tool": "idsr_check_tool",
  "args": {
    "query": "What is the patient's latest lab results?",
    "sitecode": "32060"
  }
}

When calling the "idsr_define_tool" tool, always include the following arguments in the JSON response:

- "query": the clinician's question

For example:

{
  "tool": "idsr_define_tool",
  "args": {
    "query": "What is the description of cholera?"
  }
}

There are only two cases where a tool is not needed:
1. If the clinician's question is a simple greeting, farewell, or acknowledgement.
2. The answer is clearly and completely present in the prior conversation turns.

If a tool is not needed, respond directly in natural language.

If the clinician's question or intent is ambiguous, ask a clarifying question before invoking a tool.

Never include text outside the JSON object when invoking a tool.

Never use your general knowledge to answer medical questions.

Do not reference PHI (Protected Health Information) in your responses.

Keep responses concise and focused. The clinician is a healthcare professional; do not suggest consulting one.

If the question is outside your scope, respond with "I'm sorry, I cannot assist with that request."
"""
)


builder = StateGraph(AppState)
builder.add_node(
    "assistant", lambda state: assistant(state, sys_msg, llm, llm_with_tools)
)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)


def chat_with_patient(question: str, patient_id: str, sitecode: str, thread_id: str = None):  # type: ignore
    if thread_id is None or thread_id == "":
        thread_id = str(uuid.uuid4())

    question = detect_and_redact_phi(question)["redacted_text"]

    print(question)

    # get first five characters of sitecode_selection if not none
    if sitecode is None or sitecode == "":
        sitecode_selected = ""
    else:
        sitecode_selected = sitecode[:5]

    # First turn: initialize state
    input_state: AppState = {
        "messages": [HumanMessage(content=question)],
        "pk_hash": patient_id,
        "sitecode": sitecode_selected,
    }

    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}

    output_state = react_graph.invoke(input_state, config)  # type: ignore

    for m in output_state["messages"]:
        m.pretty_print()

    assistant_message = output_state["messages"][-1].content

    # Cleaned history: Human + AI only
    chat_history_html = """
    <div style='
        border:1px solid #ccc;
        border-radius:6px;
        padding:10px;
        background-color:#f9f9f9;
        max-height:300px;
        overflow-y:auto;
    '>
    """
    for m in output_state["messages"]:
        if isinstance(m, HumanMessage):
            chat_history_html += f"<strong>You:</strong> {m.content}<br><br>"
        elif isinstance(m, AIMessage):
            chat_history_html += f"<strong>Assistant:</strong> {m.content}<br><br>"
    chat_history_html += "</div>"

    return assistant_message, thread_id, output_state.get("rag_sources", ""), "", chat_history_html

def init_session():
    new_id = str(uuid.uuid4())
    print(f"New session ID: {new_id}")
    return new_id

with gr.Blocks() as app:
    gr.Markdown(
        """
        # Clinician Assistant
        Welcome! Enter your clinical question below. The assistant can access HIV guidelines, patient data, and disease surveillance tools.

        **Note**: This is a prototype tool. There is mock data for ten fictitious patients and a mix of counties to select from (for regional variation in IDSR symptom checking).
        """
    )

    gr.Markdown("### Select Patient Context")
    with gr.Row():
        id_selected = gr.Dropdown(
            choices=[None] + [str(i) for i in range(1, 11)], label="Fake ID Number"
        )
        sitecode_selection = gr.Dropdown(
            choices=[None] + [
                "32060 - Migori",
                "32046 - Machakos",
                "32029 - Nairobi",
                "31660 - Mombasa",
                "31450 - Samburu",
            ],
            label="Sitecode",
        )

    gr.Markdown("### Ask a Clinical Question")
    question_input = gr.Textbox(label="Question")
    thread_id_state = gr.State(init_session)
    output_chat = gr.Textbox(label="Assistant Response")

    submit_btn = gr.Button("Ask")

    chat_history_display = gr.HTML()

    retrieved_sources_display = gr.HTML(label="Retrieved Sources (if applicable)")

    submit_btn.click(  # pylint: disable=no-member
        chat_with_patient,
        inputs=[question_input, id_selected, sitecode_selection, thread_id_state],
        outputs=[output_chat, thread_id_state, retrieved_sources_display, question_input, chat_history_display],
    )
    # pylint: disable=no-member
    question_input.submit(
        chat_with_patient,
        inputs=[question_input, id_selected, sitecode_selection, thread_id_state],
        outputs=[output_chat, thread_id_state, retrieved_sources_display, question_input, chat_history_display],
    )

app.launch(
    server_name="0.0.0.0",
    server_port=7860,
)
