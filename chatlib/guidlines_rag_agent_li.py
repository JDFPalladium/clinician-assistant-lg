from llama_index.core import StorageContext, load_index_from_storage
from .state_types import AppState


storage_context = StorageContext.from_defaults(persist_dir="guidance_docs/arv_metadata")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(
    similarity_top_k=3,
    similarity_threshold=0.5,
)


def rag_retrieve(query: str, llm) -> AppState:
    """Perform RAG search of repository containing authoritative information on HIV/AIDS in Kenya."""
    user_prompt = query
    sources = retriever.retrieve(user_prompt)
    retrieved_text = "\n\n".join(
        [f"Source {i+1}: {source.text}" for i, source in enumerate(sources)]
    )

    summarization_prompt = (
        "Summarize the following HIV/AIDS clinical guideline information concisely, "
        "highlighting key points relevant to the clinician's question below:\n\n"
        f"Question: {user_prompt}\n\n"
        f"Guideline Text:\n{retrieved_text}"
    )

    summary_response = llm.invoke(summarization_prompt)

    return {"rag_result": summary_response.content, "last_tool": "rag_retrieve"}  # type: ignore
