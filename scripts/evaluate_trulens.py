import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from trulens_eval import Tru
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as OpenAIFeedbackProvider
from trulens_eval.tru_app import TruLlama

# Load environment
if os.path.exists("config.env"):
    load_dotenv("config.env")

# Load vectorstore metadata
embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
df = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")

# LLMs and components
embedding_model = OpenAIEmbedding()
llm_llama = OpenAI(model="gpt-4o", temperature=0.0)
reranker = LLMRerank(llm=llm_llama, top_n=3)

# langchain summarize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

grounded = Feedback(Groundedness()).on_input().on_context().with_name("faithfulness")
context_rel = Feedback(Relevance()).on_input().on_context().with_name("context_relevance")
answer_rel = Feedback(AnswerRelevance()).on_input().on_output().with_name("answer_relevance")


# Prompt for query expansion
query_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in HIV medicine."),
    ("user", (
        "Given the query below, provide a concise, comma-separated list of related terms and synonyms "
        "useful for document retrieval. Return only the list, no explanations.\n\n"
        "Query: {query}"
    ))
])

# ---------- Functions ----------

def cosine_similarity_numpy(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm

def expand_query(query, llm, prompt_template):
    messages = prompt_template.format_messages(query=query)
    return llm.invoke(messages).content.strip()

def retrieve_contexts(expanded_query, embeddings, df, embedding_model):
    query_vec = embedding_model.get_text_embedding(expanded_query)
    sims = cosine_similarity_numpy(query_vec, embeddings)
    top_indices = sims.argsort()[-3:][::-1]
    paths = df.loc[top_indices, "vectorestore_path"].tolist()

    all_nodes = []
    for path in paths:
        ctx = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(ctx)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        all_nodes.extend(retriever.retrieve(expanded_query))

    reranked = reranker.postprocess_nodes(all_nodes, QueryBundle(expanded_query))
    return [n.text for n in reranked]

def summarize(query, contexts, llm):
    prompt = (
        "You're a clinical assistant helping a provider answer a question using HIV/AIDS guidelines.\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed summary of the most relevant points to the user question from the following source texts. Use bullet points.\n\n"
        + "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(contexts)])
    )
    return llm.invoke(prompt).content.strip()

# ---------- RAG Pipeline ----------

def custom_rag_app(query):
    expanded = expand_query(query, llm, query_expansion_prompt)
    contexts = retrieve_contexts(expanded, embeddings, df, embedding_model)
    answer = summarize(query, contexts, llm)
    return {
        "question": query,
        "expanded_query": expanded,
        "contexts": contexts,
        "answer": answer
    }


# ---------- Feedbacks ----------

provider = OpenAIFeedbackProvider()

f_grounded = Feedback(provider.groundedness).on_input().on_context().with_name("faithfulness")
f_context_rel = Feedback(provider.context_relevance).on_input().on_context().with_name("context_relevance")
f_answer_rel = Feedback(provider.relevance).on_input().on_output().with_name("answer_relevance")

# ---------- TruLens App ----------

tru_llama = TruLlama(
    app=custom_rag_app,
    feedbacks=[f_grounded, f_context_rel, f_answer_rel],
    app_id="evaluate-trulens-llama-v2"
)

tru = Tru()

# ---------- Run Evaluation ----------

test_queries = [
    "What are important drug interactions with dolutegravir?",
    "How should PrEP be provided to adolescent girls?",
    "When is cotrimoxazole prophylaxis indicated?",
    "What are the guidelines for ART failure?",
    "How do you manage HIV in pregnancy?"
]

records = []

for q in test_queries:
    record = tru_llama.run_with_record(question=q)
    fb = record["feedback"]
    records.append({
        "question": q,
        "answer": record["output"],
        "contexts": record["context"],
        "faithfulness_score": fb["faithfulness"].get("score"),
        "context_relevance_score": fb["context_relevance"].get("score"),
        "answer_relevance_score": fb["answer_relevance"].get("score"),
        "faithfulness_justification": fb["faithfulness"].get("justification", "")
    })

df = pd.DataFrame(records)
df.to_csv("trulens_llama_eval_results.csv", index=False)
print("✅ Evaluation complete. Saved to trulens_llama_eval_results.csv")
print(df)