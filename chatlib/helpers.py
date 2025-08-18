import dateparser
import dateparser.search
from datetime import datetime
from dateutil.relativedelta import relativedelta
from langchain_core.prompts import ChatPromptTemplate
import numpy as np

RELATIVE_INDICATORS = [
    "ago",
    "later",
    "before",
    "after",
    "yesterday",
    "tomorrow",
    "today",
    "tonight",
    "last",
    "next",
    "this",
    "coming",
    "previous",
    "past",
]


def is_relative_date(text_relative):
    text_lower = text_relative.lower()
    return any(word in text_lower for word in RELATIVE_INDICATORS)


def dateparser_detect(text_dates):
    results_date = dateparser.search.search_dates(text_dates, languages=["en"])
    if not results_date:
        return []
    filtered = [r for r in results_date if not is_relative_date(r[0])]
    return filtered


def describe_relative_date(dt, reference=None):
    if reference is None:
        reference = datetime.now()

    delta = relativedelta(reference, dt)

    if delta.years > 0:
        return f"{delta.years} year{'s' if delta.years > 1 else ''} ago"
    elif delta.months > 0:
        return f"{delta.months} month{'s' if delta.months > 1 else ''} ago"
    elif delta.days >= 7:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    else:
        return "today"


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