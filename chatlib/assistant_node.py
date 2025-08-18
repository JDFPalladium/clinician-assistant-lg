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
    text = "\n\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history
    )
    prompt = (
        "Summarize the clinical conversation below in a way that retains all key clinical facts and decisions.\n\n"
        f"{text}\n\nSummary:"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content


def assistant(state: AppState, sys_msg, llm, llm_with_tools) -> AppState:

    # Initialize missing keys with defaults
    state.setdefault("question", "")
    state.setdefault("pk_hash", "")
    state.setdefault("sitecode", "")
    state.setdefault("rag_result", "")
    state.setdefault("rag_sources", "")
    state.setdefault("answer", "")
    state.setdefault("last_answer", None)
    state.setdefault("last_user_message", None)
    state.setdefault("last_tool", None)
    state.setdefault("idsr_disclaimer_shown", False)
    state.setdefault("summary", None)
    state.setdefault("context", None)
    state.setdefault("context_versions", {})
    state.setdefault("last_context_injected_versions", {})
    state.setdefault("context_version_ready_for_injection", 0)
    state.setdefault("context_first_response_sent", True)

    messages = state.get("messages", [])
    base_messages = [sys_msg]
    messages = base_messages + [m for m in messages if not isinstance(m, SystemMessage)]

    # Filter out existing pk_hash and sitecode system messages and add new ones
    messages = [
        m
        for m in messages
        if not (
            isinstance(m, SystemMessage)
            and (
                m.content.startswith("Patient identifier (pk_hash):")
                or m.content.startswith("Site code:")
            )
        )
    ]

    # Inject pk_hash and sitecode as system messages if they exist and are non-empty
    pk_hash_value = state.get("pk_hash")
    if pk_hash_value:
        pk_hash_msg = SystemMessage(
            content=f"Patient identifier (pk_hash): {pk_hash_value}"
        )
        messages.append(pk_hash_msg)

    sitecode_value = state.get("sitecode")
    if sitecode_value:
        sitecode_msg = SystemMessage(content=f"Site code: {sitecode_value}")
        messages.append(sitecode_msg)

    latest_question = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    user_message_changed = latest_question != state.get("last_user_message")

    if user_message_changed:
        # Clean old tool calls before invoking new ones
        messages = remove_tool_call_messages(messages)
        state["answer"] = ""
        state["rag_result"] = ""

    # Process latest ToolMessage and update context_version
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                content = msg.content
                data = json.loads(content) if isinstance(content, str) else content

                tool_name = data.get("last_tool")
                new_context = data.get("context")

                if tool_name:
                    old_context = state.get("context", "")
                    old_version = state["context_versions"].get(tool_name, 0)

                    if new_context is not None and new_context != old_context:
                        state["context"] = new_context
                        state["context_versions"][tool_name] = old_version + 1
                        state["context_first_response_sent"] = (
                            False  # Reset flag on new context
                        )

                    state["last_tool"] = tool_name

                for k, v in data.items():
                    if k not in ("context", "last_tool"):
                        state[k] = v

                break
            except json.JSONDecodeError:
                break

    tool_name = "idsr_check"
    current_version = state["context_versions"].get(tool_name, 0)
    last_injected_version = state["last_context_injected_versions"].get(tool_name, 0)

    # On turns where user message is unchanged, advance ready_for_injection to current_version
    if (
        not user_message_changed
        and state["context_version_ready_for_injection"] < current_version
    ):
        state["context_version_ready_for_injection"] = current_version

    # Inject context system message only if:
    # - last_tool matches tool_name
    # - context exists
    # - ready_for_injection > last injected version
    # - AND first AI response after new context has been sent
    if (
        state.get("last_tool") == tool_name
        and state.get("context")
        and state["context_version_ready_for_injection"] > last_injected_version
        and state.get("context_first_response_sent", True)
    ):
        context_msg = SystemMessage(
            content=(
                f"The following information was retrieved from the {tool_name.upper()} database and may help answer the user's question:\n\n"
                f"{state['context']}\n\n"
                "Use this information when responding."
            )
        )
        messages.append(context_msg)

        state["last_context_injected_versions"][tool_name] = state[
            "context_version_ready_for_injection"
        ]
        state["last_tool"] = None

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

    else:
        final_content = new_message.content

    # Add disclaimer if needed
    if state.get("last_tool") == "idsr_check" and not state.get(
        "idsr_disclaimer_shown", False
    ):
        disclaimer = (
            "Disclaimer: This is not a diagnosis. This is meant to help "
            "identify possible matches based on priority IDSR diseases for clinician awareness.\n\n"
        )
        final_content = disclaimer + final_content
        state["idsr_disclaimer_shown"] = True

    # After generating AI message, mark first response sent
    if (
        state.get("last_tool") == tool_name
        or state.get("context_first_response_sent") is False
    ):
        state["context_first_response_sent"] = True

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
    human_ai_messages = [
        m for m in non_sys_messages if isinstance(m, (HumanMessage, AIMessage))
    ]

    if len(human_ai_messages) > 10:
        summary_text = summarize_conversation(messages, llm)
        summary_msg = SystemMessage(
            content="Summary of earlier conversation:\n" + summary_text
        )

        # Keep sys_msg, the new summary message, and the last 5 Human/AI messages
        recent_msgs = [
            m for m in reversed(messages) if isinstance(m, (HumanMessage, AIMessage))
        ][:5]
        recent_msgs.reverse()
        messages = [sys_msg, summary_msg] + recent_msgs

    state["answer"] = final_content
    state["messages"] = messages
    state["last_user_message"] = latest_question
    state["question"] = latest_question

    return state
