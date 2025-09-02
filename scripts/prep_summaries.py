import numpy as np
import pandas as pd
from llama_index.embeddings.openai import OpenAIEmbedding
import os
import csv
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def prep_summaries(
    base_dir="data/processed/lp/indices",
    output_csv="data/raw/guidelines_summaries.csv",
    output_embeddings="data/processed/lp/summary_embeddings/embeddings.npy",
    output_index="data/processed/lp/summary_embeddings/index.tsv",
    model_name="gpt-4o",
    temperature=0.0,
):
    load_dotenv("config.env")
    llm = ChatOpenAI(temperature=temperature, model=model_name)

    # collect all subfolders under indices (excluding "Global")
    chapter_folders = [
        os.path.join(base_dir, f) for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f != "Global"
    ]

    PROMPT_TEMPLATE = """
    You are preparing a summary of a clinical guideline chapter for use in a retrieval-augmented generation (RAG) system.

    Your goal is to create a summary that:
    - Clearly differentiates this chapter from others in the same HIV guideline collection
    - Lists all major clinical topics, drugs, diseases, populations, and interventions discussed
    - Includes synonyms, abbreviations, and alternate terms clinicians might use
    - Mentions any unique recommendations, exceptions, or special scenarios covered in this chapter
    - Avoids copying large blocks of text, but is more detailed than a generic overview

    Write the summary as a concise paragraph or a bulleted list. Include enough detail and keywords so that a search for any relevant clinical question about this chapter would match this summary.

    Chapter text:
    \"\"\"{chapter_text}\"\"\"
    """

    all_summaries = []

    for folder in chapter_folders:
        print(f"Loading nodes from {folder} ...")
        sc = StorageContext.from_defaults(persist_dir=folder)
        idx = load_index_from_storage(sc)
        nodes = list(idx.storage_context.docstore.docs.values())
        chapter_text = "\n".join([node.text for node in nodes])
        summary = llm.invoke(PROMPT_TEMPLATE.format(chapter_text=chapter_text))
        all_summaries.append(summary)

    # Write out summaries to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["vectorstore_path", "summary"])
        for i, summary in enumerate(all_summaries):
            writer.writerow([chapter_folders[i], summary.content])

    # Load the chapter summaries and embed
    df = pd.read_csv(output_csv)
    embedding_model = OpenAIEmbedding()
    summary_embeddings = []

    for summary in df["summary"]:
        emb = embedding_model.get_text_embedding(summary)
        summary_embeddings.append(emb)

    summary_embeddings = np.vstack(summary_embeddings)

    # Save embeddings and metadata
    os.makedirs(os.path.dirname(output_embeddings), exist_ok=True)
    np.save(output_embeddings, summary_embeddings)
    df.to_csv(output_index, sep="\t", index=False)

    print("✅ Saved embeddings and index.")

# Example usage:
if __name__ == "__main__":
    prep_summaries()