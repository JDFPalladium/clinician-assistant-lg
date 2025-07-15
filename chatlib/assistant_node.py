from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from chatlib.state_types import AppState
import json

def remove_tool_call_messages(messages):
    new_messages = []
    skip_tool_call_ids = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                skip_tool_call_ids.add(call["id"])
            continue  # skip AIMessage with tool calls
        if isinstance(msg, ToolMessage) and msg.tool_call_id in skip_tool_call_ids:
            continue  # skip ToolMessages corresponding to removed AIMessage
        new_messages.append(msg)
    return new_messages

def summarize_conversation(messages, llm):
    """Summarizes the conversation history (excluding system messages)."""
    history = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    text = "\n\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in history)
    prompt = (
        "Summarize the clinical conversation below in a way that retains all key clinical facts and decisions.\n\n"
        f"{text}\n\nSummary:"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content


def assistant(state: AppState, sys_msg, llm, llm_with_tools) -> AppState:
    messages = state.get("messages", [])
    base_messages = [sys_msg]
    messages = base_messages + [m for m in messages if not isinstance(m, SystemMessage)]

    latest_question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    user_message_changed = latest_question != state.get("last_user_message")

    if user_message_changed:
        # Clean old tool calls before invoking new ones
        messages = remove_tool_call_messages(messages)
        state["answer"] = ""
        state["rag_result"] = ""

    # Update state from any ToolMessages appended by previous tool calls
    # Only consider the most recent ToolMessage for updating state
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                content = msg.content
                data = json.loads(content) if isinstance(content, str) else content
                state.update(data)
                break  # only process the most recent ToolMessage
            except json.JSONDecodeError:
                break


    # Invoke LLM with tools (this returns AIMessage with tool_calls if tool call is needed)
    new_message = llm_with_tools.invoke(messages)
    messages.append(new_message)
  
    # If the new_message has tool_calls, it means a tool call is pending; return now so tool node runs
    if getattr(new_message, "tool_calls", None):
        state["messages"] = messages
        state["last_user_message"] = latest_question
        return state

    # No more tool calls: generate final answer from state or AIMessage content
    if state.get("answer"):
        final_content = state["answer"]

    elif state.get("rag_result"):
        # Use conversation history + a system message to inject RAG guidance
        rag_msg = SystemMessage(
            content="The following clinical guidelines may help answer the user's question:\n\n"
                    f"{state['rag_result']}\n\n"
                    "Use this information when responding."
        )
        messages_with_rag = messages + [rag_msg]
        llm_response = llm.invoke(messages_with_rag)
        final_content = llm_response.content

    else:
        final_content = new_message.content

    # Add disclaimer if needed
    if state.get("last_tool") == "idsr_check" and not state.get("idsr_disclaimer_shown", False):
        disclaimer = (
            "Disclaimer: This is not a diagnosis. This is meant to help "
            "identify possible matches based on priority IDSR diseases for clinician awareness.\n\n"
        )
        final_content = disclaimer + final_content
        state["idsr_disclaimer_shown"] = True

    # Replace the last AIMessage content with final_content to avoid duplicates
    for i in reversed(range(len(messages))):
        if isinstance(messages[i], AIMessage):
            messages[i] = AIMessage(content=final_content)
            break
    else:
        # fallback: append if no AIMessage found (rare)
        messages.append(AIMessage(content=final_content))

    # Summarization logic 
    non_sys_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    human_ai_messages = [m for m in non_sys_messages if isinstance(m, (HumanMessage, AIMessage))]

    if len(human_ai_messages) > 15:
        summary_text = summarize_conversation(messages, llm)
        summary_msg = SystemMessage(content="Summary of earlier conversation:\n" + summary_text)

        # Keep sys_msg, the new summary message, and the last 5 Human/AI messages
        recent_msgs = [m for m in reversed(messages) if isinstance(m, (HumanMessage, AIMessage))][:5]
        recent_msgs.reverse()
        messages = [sys_msg, summary_msg] + recent_msgs

    state["answer"] = final_content
    state["messages"] = messages
    state["last_user_message"] = latest_question
    state["question"] = latest_question

    return state
