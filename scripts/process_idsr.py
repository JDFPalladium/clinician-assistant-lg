import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from rapidfuzz import fuzz

# === Setup ===
base_dir = os.path.dirname(__file__)
raw_path = os.path.abspath(os.path.join(base_dir, "data", "raw"))
processed_path = os.path.abspath(os.path.join(base_dir, "data", "processed"))
os.makedirs(processed_path, exist_ok=True)

load_dotenv(os.path.join(base_dir, "config.env"))
api_key = os.environ.get("OPENAI_API_KEY")

# === Step 1: Read IDSR Text ===
with open(os.path.join(raw_path, "IDSR.txt"), encoding="utf-8") as f:
    text = f.read()

# === Step 2: Extract Keywords via GPT ===
prompt = """
You are a helpful assistant. Extract a list of 30–50 key symptoms, signs, or diagnostic terms from the following disease descriptions.

Focus on words or phrases that are likely to appear in clinical case definitions or user queries — such as "fever", "skin lesions", "swollen lymph nodes", "positive blood smear", etc.

Only return the keywords or short phrases — one per line.

Text:
"""

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt + text}
    ],
    temperature=0.0
)

# Normalize keywords
keywords = [line.strip() for line in response.choices[0].message.content.splitlines() if line.strip()]
def normalize_kw(kw):
    return kw.lstrip("-• ").strip().lower()
keywords = [normalize_kw(kw) for kw in keywords]

# Save keywords
kw_path = os.path.join(processed_path, "idsr_keywords.txt")
with open(kw_path, "w", encoding="utf-8") as f:
    for keyword in keywords:
        f.write(f"{keyword}\n")

print(f"✅ Saved keywords to {kw_path}")

# === Step 3: Parse Disease Sections ===
def parse_disease_text(text):
    diseases = []
    lines = text.strip().splitlines()

    current_disease = None
    current_subsection = None
    buffer = []

    def finalize_subsection():
        if current_disease is not None and current_subsection and buffer:
            content = " ".join(line.strip() for line in buffer).strip()
            current_disease[current_subsection] = content

    subsection_pattern = re.compile(r"^-\s*(.+):\s*$")

    for line in lines + [""]:
        if not line.strip():
            finalize_subsection()
            if current_disease:
                diseases.append(current_disease)
            current_disease = None
            current_subsection = None
            buffer = []
            continue

        if current_disease is None:
            current_disease = {"disease_name": line.strip()}
            continue

        match = subsection_pattern.match(line)
        if match:
            finalize_subsection()
            current_subsection = match.group(1).strip()
            buffer = []
        else:
            buffer.append(line.rstrip())

    return diseases

disease_dicts = parse_disease_text(text)

# === Step 4: Convert to LangChain Documents ===
def convert_disease_dicts_to_documents(disease_dicts):
    docs = []
    for disease in disease_dicts:
        disease_name = disease.get("disease_name", "")
        subsections = [f"{key}:\n{value}" for key, value in disease.items() if key != "disease_name"]
        full_text = f"Disease: {disease_name}\n\n" + "\n\n".join(subsections)
        docs.append(Document(page_content=full_text, metadata={"disease_name": disease_name}))
    return docs

documents = convert_disease_dicts_to_documents(disease_dicts)

# === Step 5: Tag Documents with Keywords ===
def tag_documents_with_keywords(documents, keywords, threshold=85):
    tagged = []
    for doc in documents:
        content = doc.page_content.lower()
        matched = [kw for kw in keywords if fuzz.partial_ratio(kw.lower(), content) >= threshold]
        doc.metadata["matched_keywords"] = matched
        tagged.append(doc)
    return tagged

tagged_documents = tag_documents_with_keywords(documents, keywords)

# Save JSON version
json_path = os.path.join(processed_path, "tagged_documents.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump([doc.dict() for doc in tagged_documents], f, ensure_ascii=False, indent=2)

print(f"✅ Saved tagged documents to {json_path}")

# === Step 6: Build and Save FAISS Vectorstore ===
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(tagged_documents, embedding_model)
vs_path = os.path.join(processed_path, "disease_vectorstore")
vectorstore.save_local(vs_path)

print(f"✅ Saved FAISS vectorstore to {vs_path}")
