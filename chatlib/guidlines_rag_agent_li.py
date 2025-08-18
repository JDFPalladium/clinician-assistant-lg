from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from .state_types import AppState
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import pandas as pd
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI

# try hybrid with hierarchical search and flat search
storage_context_arv = StorageContext.from_defaults(persist_dir="data/processed/arv_metadata")
index_arv = load_index_from_storage(storage_context_arv)
arv_retriever = VectorIndexRetriever(index=index_arv, similarity_top_k=3)

# load vectorstore summaries
embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
df = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")

embedding_model = OpenAIEmbedding()

# Define your reranker-compatible LLM
llm_llama = OpenAI(model="gpt-4o", temperature=0.0)

# Create LLM reranker
reranker = LLMRerank(llm=llm_llama, top_n=3)

# summarization LLM
summarizer_llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0125")

# Define a prompt template for query expansion
query_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in HIV medicine."),
    ("user", (
        "Given the query below, provide a concise, comma-separated list of related terms and synonyms "
        "useful for document retrieval. Return only the list, no explanations.\n\n"
        "Query: {query}"
    ))
])

def expand_query(query: str, llm) -> str:
    messages = query_expansion_prompt.format_messages(query=query)
    response = llm.invoke(messages)
    expanded = response.content.strip()
    # If output is multiline list, convert to comma-separated string
    if "\n" in expanded:
        lines = [line.strip("- ").strip() for line in expanded.splitlines() if line.strip()]
        expanded = ", ".join(lines)
    print(f"Expanded query: {expanded}")
    return expanded

def cosine_similarity_numpy(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Normalize the query vector and the matrix
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    
    # Dot product gives cosine similarity
    return matrix_norm @ query_norm

def cosine_rerank(query_vec, nodes, embedder, top_n=3):
    texts = [n.text for n in nodes]
    node_vecs = embedder.get_text_embedding_batch(texts)
    sims = cosine_similarity_numpy(query_vec, np.array(node_vecs))
    top_idxs = sims.argsort()[-top_n:][::-1]
    return [nodes[i] for i in top_idxs]

def format_sources_for_html(sources):
    html_blocks = []
    for i, source in enumerate(sources):
        text = source.text.replace("\n", "<br>").strip()
        block = f"""
        <details style='margin-bottom: 1em;'>
            <summary><strong>Source {i+1}</strong></summary>
            <div style='margin-top: 0.5em; font-family: monospace;'>{text}</div>
        </details>
        """
        html_blocks.append(block)
    return "\n".join(html_blocks)


def rag_retrieve(query: str, llm) -> AppState:
    """Perform RAG search of repository containing authoritative information on HIV/AIDS in Kenya."""
    
    # Step 1: Expand the user query
    # query_bundle = QueryBundle(query) # use original query for reranking
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
        raw_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
        sources_raw = raw_retriever.retrieve(expanded_query)
        all_sources.extend(sources_raw)

    # now, let's also load in three chunks from general db
    sources_arv = arv_retriever.retrieve(expanded_query)
    all_sources.extend(sources_arv)

    # Run retrieval (vector search) and reranking manually
    print(f"Retrieved {len(all_sources)} raw sources from vector search.")
    # sources = reranker.postprocess_nodes(all_sources, query_bundle)
    sources = cosine_rerank(query_embedding, all_sources, embedding_model, top_n=2)
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
    
    
    summarization_prompt = (
        "You're a clinical assistant helping a provider answer a question using HIV/AIDS guidelines.\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed summary of the most relevant points to the user question from the following source texts and use bullet points. \n\n"
        # "If the sources do not contain relevant information, simply say 'No relevant information found in the sources.'\n\n"
        f"{retrieved_text}"
    )

    print("Prompt length in characters:", len(summarization_prompt))
    summary_response = summarizer_llm.invoke(summarization_prompt)

    return {"rag_result": summary_response.content,
            "rag_sources": format_sources_for_html(sources),
            "last_tool": "rag_retrieve"
        }  # type: ignore
