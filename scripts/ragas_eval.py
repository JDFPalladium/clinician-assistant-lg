# custom_rag_with_ragas.py

import numpy as np
import pandas as pd
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
if os.path.exists("config.env"):
    load_dotenv("config.env")

embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
df = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")

embedding_model = OpenAIEmbedding()

# Define your reranker-compatible LLM
llm_llama = OpenAI(model="gpt-4o", temperature=0.0)

# Create LLM reranker
reranker = LLMRerank(llm=llm_llama, top_n=3)

# summarizer LLM
llm = ChatOpenAI(temperature=0.0, model="gpt-4o")

# Define a prompt template for query expansion
query_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in HIV medicine."),
    ("user", (
        "Given the query below, provide a concise, comma-separated list of related terms and synonyms "
        "useful for document retrieval. Return only the list, no explanations.\n\n"
        "Query: {query}"
    ))
])

def cosine_similarity_numpy(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


def expand_query(query, llm, prompt_template):
    messages = prompt_template.format_messages(query=query)
    return llm.invoke(messages).content.strip()

def retrieve_contexts(expanded_query, embeddings, df, embedding_model):
    query_vec = embedding_model.get_text_embedding(expanded_query)
    similarities = cosine_similarity_numpy(query_vec, embeddings)
    top_indices = similarities.argsort()[-3:][::-1]
    paths = df.loc[top_indices, "vectorestore_path"].tolist()
    print(paths)
    all_nodes = []
    for path in paths:
        ctx = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(ctx)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        all_nodes.extend(retriever.retrieve(expanded_query))
    
    return [n.text for n in LLMRerank(llm=llm_llama, top_n=3).postprocess_nodes(all_nodes, QueryBundle(expanded_query))]

def summarize(query, contexts, llm):
    prompt = (
        "You're a clinical assistant helping a provider answer a question using HIV/AIDS guidelines.\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed summary of the most relevant points from the following source texts using bullet points.\n\n"
        + "\n\n".join([f"Source {i+1}: {text}" for i, text in enumerate(contexts)])
    )
    return llm.invoke(prompt).content.strip()

# Run on test queries
test_queries = [
    "What are important drug interactions with dolutegravir?",
    "How should PrEP be provided to adolescent girls?",
    "When is cotrimoxazole prophylaxis indicated?",
    "What are the guidelines for ART failure?",
    "How do you manage HIV in pregnancy?"
]
results = []

for q in test_queries:
    print(f"⏳ Processing: {q}")
    expanded = expand_query(q, llm, query_expansion_prompt)
    contexts = retrieve_contexts(expanded, embeddings, df, embedding_model)
    answer = summarize(q, contexts, llm)
    results.append({
        "question": q,
        "contexts": contexts,
        "answer": answer
    })

# --- Ragas Evaluation ---
print("✅ Running Ragas evaluation...")

ragas_data = Dataset.from_list(results)

eval_results = evaluate(
    ragas_data,
    metrics=[faithfulness, answer_relevancy]
)

df_eval = eval_results.to_pandas()
df_eval.to_csv("ragas_eval_results.csv", index=False)

print("✅ Evaluation complete. Saved to ragas_eval_results.csv")
print(df_eval)
