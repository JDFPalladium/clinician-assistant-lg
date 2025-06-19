from llama_index.core import StorageContext, load_index_from_storage
from .state_types import State

# Load index for retrieval
storage_context = StorageContext.from_defaults(persist_dir="arv_metadata")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=5,
                                # Similarity threshold for filtering
                                similarity_threshold=0.5)

def rag_retrieve(state:State) -> State:
    """Perform RAG search of repository containing authoritative information on HIV/AIDS in Kenya.

    """
    user_prompt = state["question"]  # or whatever key holds the prompt
    sources = retriever.retrieve(user_prompt)
    retrieved_text = "\n\n".join([f"Source {i+1}: {source.text}" for i, source in enumerate(sources)])

    return {**state, "rag_result": "RAG search results for: " + retrieved_text}

if __name__ == "__main__":
    # Test the function
    test_state = State(
        messages=[],
        question="What are the first-line treatments for HIV in Kenya?",
        rag_result="",
        query="",
        result="",
        answer=""
    )
    updated_state = rag_retrieve(test_state)
    print(updated_state["rag_result"])