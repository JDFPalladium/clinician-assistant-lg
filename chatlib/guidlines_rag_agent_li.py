from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from .state_types import AppState
import numpy as np
import pandas as pd
from llama_index.embeddings.openai import OpenAIEmbedding
from .helpers import expand_query, cosine_similarity_numpy, cosine_rerank, format_sources_for_html

# load vectorstore summaries
embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
df = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")

embedding_model = OpenAIEmbedding()

# Define your reranker-compatible LLM
llm_llama = OpenAI(model="gpt-4o", temperature=0.0)

# Create LLM reranker
reranker = LLMRerank(llm=llm_llama, top_n=2)

def rag_retrieve(query: str, llm, global_retriever) -> AppState:
    """Perform RAG search of repository containing authoritative information on HIV/AIDS in Kenya."""
    
    # Step 1: Expand the user query
    query_bundle = QueryBundle(query) # use original query for reranking
    expanded_query = expand_query(query, llm)

    # Embed the expanded query and find similar summaries
    query_embedding = embedding_model.get_text_embedding(expanded_query)
    similarities = cosine_similarity_numpy(query_embedding, embeddings)
    top_indices = similarities.argsort()[-3:][::-1]
    selected_paths = df.loc[top_indices, "vectorestore_path"].tolist()
    print(f"Selected paths for retrieval: {selected_paths}")

    # For each path in selected paths, load the index and retrieve documents
    all_sources = []
    for path in selected_paths:
        storage_context = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(storage_context)
        raw_retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        sources_raw = raw_retriever.retrieve(expanded_query)
        all_sources.extend(sources_raw)

    # now, let's also load in three chunks from general db
    sources_arv = global_retriever.retrieve(expanded_query)
    all_sources.extend(sources_arv)
    print(f"{len(all_sources)} sources before deduplication.")

    # --- Deduplicate by node_id ---
    unique_sources = {}
    for src in all_sources:
        node_id = src.node.node_id
        # keep the one with the higher score if duplicate
        if node_id not in unique_sources or src.score > unique_sources[node_id].score:
            unique_sources[node_id] = src

    deduped_sources = list(unique_sources.values())
    print(f"{len(deduped_sources)} sources remain after deduplication.")

    # Run retrieval (vector search) and reranking manually
    print(f"Retrieved {len(deduped_sources)} raw sources from vector search.")
    sources = reranker.postprocess_nodes(deduped_sources, query_bundle)
    # sources = cosine_rerank(query_embedding, deduped_sources, embedding_model, top_n=2)
    print(f"Retrieved {len(sources)} sources after reranking.")

    if not sources:
        return {
            "rag_result": "No relevant information found in the sources. Please try rephrasing your question.",
            "last_tool": "rag_retrieve"
        }
    # Format the retrieved sources for the response (and remove lengthy white space or repeated dashes)
    retrieved_text = "\n\n".join([
        f"Source {i+1}: {source.text}" for i, source in enumerate(sources)
    ])
    
    # Use conversation history + a system message to inject RAG guidance
    prompt = (
        "Based on the following clinical guideline excerpts, answer the clinician's question as precisely as possible.\n\n"
        "Focus only on information that directly addresses the question.\n"
        "Do not include information not explicitly contained in the sources.\n"
        "Do not include background or general recommendations unless they are explicitly relevant.\n"
        "If the information is not present in the sources, do not make assumptions or provide your own interpretations.\n\n"
        f"Clinician question: {query}\n\n"
        "Guideline excerpts:\n"
        f"{retrieved_text}"
    )

    response = llm.invoke(prompt)
    answer_text = response.content

    return {"answer": answer_text,
            "rag_sources": format_sources_for_html(sources),
            "last_tool": "rag_retrieve"
        }  # type: ignore
