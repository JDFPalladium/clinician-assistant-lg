from llama_index.core import StorageContext, load_index_from_storage
from langchain_core.tools import tool
from .state_types import AppState
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")

# Load index for retrieval
storage_context = StorageContext.from_defaults(persist_dir="arv_metadata")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=3,
                                # Similarity threshold for filtering
                                similarity_threshold=0.5)

@tool
def rag_retrieve(state:AppState) -> AppState:
    """Perform RAG search of repository containing authoritative information on HIV/AIDS in Kenya.

    """
    user_prompt = state["question"]  # or whatever key holds the prompt
    sources = retriever.retrieve(user_prompt)
    retrieved_text = "\n\n".join([f"Source {i+1}: {source.text}" for i, source in enumerate(sources)])
    
    summarization_prompt = (
    "Summarize the following HIV/AIDS clinical guideline information concisely, "
    "highlighting key points relevant to the clinician's question below:\n\n"
    f"Question: {user_prompt}\n\n"
    f"Guideline Text:\n{retrieved_text}"
    )

    # Call your LLM to generate the summary
    summary_response = llm.invoke(summarization_prompt)

    # Store the summary in state instead of full retrieved text
    state['rag_result'] = summary_response.content

    return state
    # return {**state, "rag_result": "RAG search results for: " + retrieved_text}

# if __name__ == "__main__":
#     # Test the function
#     test_state = AppState(
#         messages=[],
#         question="What are the first-line treatments for HIV in Kenya?",
#         rag_result="",
#         query="",
#         result="",
#         answer=""
#     )
#     updated_state = rag_retrieve(test_state)
#     print(updated_state["rag_result"])