# expanded_ragas_eval.py

import os
import time
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

if os.path.exists("config.env"):
    load_dotenv("config.env")

# Load embeddings for prefiltering top documents
embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
df_summaries = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")

embedding_model = OpenAIEmbedding()

# Prompt for query expansion
query_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in HIV medicine."),
    ("user", "Given the query below, provide a concise, comma-separated list of related terms and synonyms useful for document retrieval. Return only the list, no explanations.\n\nQuery: {query}")
])

def cosine_similarity_numpy(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm

def expand_query(query, llm):
    messages = query_expansion_prompt.format_messages(query=query)
    return llm.invoke(messages).content.strip()

def retrieve(expanded_query):
    query_vec = embedding_model.get_text_embedding(expanded_query)
    sims = cosine_similarity_numpy(query_vec, embeddings)
    top_paths = df_summaries.loc[sims.argsort()[-3:][::-1], "vectorestore_path"].tolist()

    all_nodes = []
    for path in top_paths:
        ctx = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(ctx)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        all_nodes.extend(retriever.retrieve(expanded_query))
    return all_nodes

def cosine_rerank(query_vec, nodes, embedder, top_n=3):
    texts = [n.text for n in nodes]
    node_vecs = embedder.get_text_embedding_batch(texts)
    sims = cosine_similarity_numpy(query_vec, np.array(node_vecs))
    top_idxs = sims.argsort()[-top_n:][::-1]
    return [nodes[i] for i in top_idxs]

def hybrid_rerank(query_vec, nodes, embedder, llm_reranker, expanded, top_n_cosine=5, top_n_llm=2):
    texts = [n.text for n in nodes]
    node_vecs = embedder.get_text_embedding_batch(texts)
    sims = cosine_similarity_numpy(query_vec, np.array(node_vecs))
    top_idxs = sims.argsort()[-top_n_cosine:][::-1]
    prefiltered_nodes = [nodes[i] for i in top_idxs]
    bundle = QueryBundle(expanded)
    reranked_nodes = llm_reranker.postprocess_nodes(prefiltered_nodes, bundle)
    return reranked_nodes[:top_n_llm]

def summarize(query, contexts, llm):
    prompt = (
        "You're a clinical assistant helping a provider answer a question using HIV/AIDS guidelines.\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed summary of the most relevant points from the following source texts using bullet points.\n\n"
        + "\n\n".join([f"Source {i+1}: {text}" for i, text in enumerate(contexts)])
    )
    return llm.invoke(prompt).content.strip()

def generate_final_response(query, summary, llm):
    prompt = (
        "Based on the following clinical guideline excerpts, answer the clinician's question as precisely as possible.\n\n"
        "Focus only on information that directly addresses the question.\n"
        "Do not include background or general recommendations unless they are explicitly relevant.\n\n"
        "Guideline excerpts:\n"
        f"Clinician question:\n{query}\n\n"
        f"Guideline summary:\n{summary}\n"
    )
    return llm.invoke(prompt).content.strip()

def run_eval(test_queries, reranker_type="llm", summarizer_model="gpt-4o"):
    results = []
    expander_llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
    summarizer_llm = ChatOpenAI(temperature=0.0, model=summarizer_model)
    reranker_llm = OpenAI(model="gpt-4o")
    reranker = LLMRerank(llm=reranker_llm, top_n=3)

    for query in test_queries:
        print(f"⏳ {query} [{reranker_type}, {summarizer_model}]")
        start_time = time.time()

        expanded = expand_query(query, expander_llm)
        print(expanded)
        query_vec = embedding_model.get_text_embedding(expanded)
        retrieved_nodes = retrieve(expanded)

        if reranker_type == "llm":
            reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, QueryBundle(expanded))[:3]
        elif reranker_type == "cosine":
            reranked_nodes = cosine_rerank(query_vec, retrieved_nodes, embedding_model, top_n=3)
        elif reranker_type == "hybrid":
            reranked_nodes = hybrid_rerank(query_vec, retrieved_nodes, embedding_model, reranker, expanded=expanded, top_n_cosine=5, top_n_llm=2)
        else:
            raise ValueError("Invalid reranker type")

        if not reranked_nodes:
            print(f"⚠️ Skipping summarization for: {query} (no retrieved sources)")
            results.append({
                "question": query,
                "contexts": [],
                "answer": "No relevant information found in the sources.",
                "response_time": round(time.time() - start_time, 2)
            })
            continue

        summary = summarize(query, [n.text for n in reranked_nodes], summarizer_llm)
        answer = generate_final_response(query, summary, expander_llm)
        total_time = time.time() - start_time

        results.append({
            "question": query,
            "contexts": [n.text for n in reranked_nodes],
            "answer": answer,
            "response_time": round(total_time, 2)
        })

    ragas_data = Dataset.from_list(results)
    eval_results = evaluate(ragas_data, metrics=[faithfulness, answer_relevancy])
    df_eval = eval_results.to_pandas()
    df_eval["response_time"] = [r["response_time"] for r in results]
    return df_eval

# Run 6 configs
test_queries = [
    "What are important drug interactions with dolutegravir?",
    "How should PrEP be provided to adolescent girls?",
    "When is cotrimoxazole prophylaxis indicated?",
    "What are the guidelines for ART failure?",
    "How do you manage HIV in pregnancy?",
    "When should infants start ART?",
    "What is the recommended PrEP regimen for men who have sex with men?",
    "How often should viral load be monitored?",
    "What is the preferred first-line regimen for adults?",
    "Can pregnant women use dolutegravir?",
    "When is tenofovir not recommended?",
    "How should HIV be managed in tuberculosis coinfection?",
    "What lab tests are used to monitor ART?",
    "When is second-line ART initiated?",
    "What adherence strategies are recommended?",
    "What are the contraindications to efavirenz?",
    "Can HIV be managed with a two-drug regimen?",
    "How do you handle treatment failure?",
    "When is regimen switching appropriate?",
    "What is the role of resistance testing?"
]

combinations = [
    ("llm", "gpt-4o"),
    ("llm", "gpt-3.5-turbo-0125"),
    # ("hybrid", "gpt-4o"),
    # ("hybrid", "gpt-3.5-turbo-0125"),
    ("cosine", "gpt-4o"),
    ("cosine", "gpt-3.5-turbo-0125"),
]

all_dfs = []
for reranker_type, summarizer_model in combinations:
    df = run_eval(test_queries, reranker_type, summarizer_model)
    df["reranker"] = reranker_type
    df["summarizer"] = summarizer_model
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("ragas_eval_full_combinations.csv", index=False)
print("✅ All evaluations complete. Saved to ragas_eval_full_combinations.csv")
