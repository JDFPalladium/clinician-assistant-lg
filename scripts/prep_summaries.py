import numpy as np
import pandas as pd
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv
load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

# load vectorstore summaries
df = pd.read_csv("data/raw/guidelines_summaries.csv")

# Embed summaries
embedding_model = OpenAIEmbedding()
summary_embeddings = []

for summary in df["summary"]:
    emb = embedding_model.get_text_embedding(summary)
    summary_embeddings.append(emb)

summary_embeddings = np.vstack(summary_embeddings)

# Save embeddings and metadata
os.makedirs("data/processed/lp/summary_embeddings", exist_ok=True)

np.save("data/processed/lp/summary_embeddings/embeddings.npy", summary_embeddings)
df.to_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t", index=False)

print("✅ Saved embeddings and index.")