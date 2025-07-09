from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from chatlib.state_types import AppState

# Assistant Node
def assistant(state: AppState, sys_msg, llm, llm_with_tools) -> AppState:
    
    pk_hash = state.get("pk_hash", None)

    if pk_hash:
        pk_msg = SystemMessage(content=f"The patient identifier (pk_hash) is: {pk_hash}")
        messages = [sys_msg, pk_msg] + state.get("messages", [])
    else:
        messages = [sys_msg] + state.get("messages", [])

    # Extract latest human question
    latest_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_question = msg.content
            break

    # Generate AIMessage only if answer is new
    if "answer" in state and state["answer"]:
        if state.get("last_answer") != state["answer"]:
            last_tool = state.get("last_tool")

            if last_tool == "idsr_check":
                # Formatting instructions for idsr_check
                format_instructions = """
Please format the following medical assistant response exactly as:

Likely matches:
- Disease Name: [Likely] – Reason
- Disease Name: [Probable] – Reason
(Only include diseases that clearly fit based on the information.)

If none:
- No strong match found.

Clarifying questions (optional, only if needed):
- Question 1
- Question 2

At the end, always give a brief recommendation like:
- Recommendation: "Suggest monitoring for the listed conditions." OR "No disease meets criteria based on current data — suggest gathering additional history on [x, y, z]."
"""

                # Combine formatting instructions with raw answer
                prompt = f"{format_instructions}\n\nResponse:\n{state['answer']}"

                # Call LLM to reformat the answer
                llm_response = llm.invoke(prompt)
                formatted_answer = llm_response.content.strip()

                ai_message = AIMessage(content=formatted_answer)
            else:
                # For other tools, use the raw answer as is
                ai_message = AIMessage(content=state["answer"])

            messages = messages + [ai_message]
            state["messages"] = messages
            state["question"] = latest_question
            state["last_answer"] = state["answer"]  # track processed answer
            return state

    # Otherwise, normal LLM with tools invocation
    new_message = llm_with_tools.invoke(messages)
    messages = messages + [new_message]
    state["messages"] = messages
    state["question"] = latest_question
    return state