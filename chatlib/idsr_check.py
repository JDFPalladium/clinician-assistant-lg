from .state_types import AppState
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import json
import math
from collections import Counter

## Keywords
# load keywords from file
with open("./guidance_docs/idsr_keywords.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]

# load vectorstore
vectorstore = FAISS.load_local(
    "./guidance_docs/disease_vectorstore",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

# load tagged documents from JSON for keyword matching to document metadata
with open("./guidance_docs/tagged_documents.json", "r", encoding="utf-8") as f:
    doc_dicts = json.load(f)

tagged_documents = [Document(**d) for d in doc_dicts]

# Set up metrics for keywords
# Count how many documents each keyword appears in
keyword_doc_counts = Counter()
total_docs = len(tagged_documents)

for doc in tagged_documents:
    seen = set(doc.metadata.get("matched_keywords", []))
    for kw in seen:
        keyword_doc_counts[kw] += 1

# Use log-scaled inverse frequency to avoid extreme values
keyword_weights = {
    kw: math.log(total_docs / (1 + count))  # add 1 to avoid div-by-zero
    for kw, count in keyword_doc_counts.items()
}


def score_doc(doc_to_score, matched_keywords):
    doc_keywords = set(doc_to_score.metadata.get("matched_keywords", []))
    overlap = doc_keywords & set(matched_keywords)
    return sum(keyword_weights.get(kw, 0) for kw in overlap)


## Define helper functions
class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(
        description="List of relevant keywords extracted from the query"
    )


def extract_keywords_with_gpt(query: str, llm, known_keywords: List[str]) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=KeywordsOutput)

    prompt = ChatPromptTemplate.from_template(
        """
You are helping identify relevant medical concepts. 
Given this query: "{query}"

Select the most relevant 3-5 keywords from this list:
{keyword_list}

Return the matching keywords as a JSON object with a single key "keywords" whose value is a list of strings.

{format_instructions}
"""
    )

    # Compose the chain as a RunnableSequence: prompt -> llm -> parser
    chain = prompt | llm | parser

    output = chain.invoke(
        {
            "query": query,
            "keyword_list": ", ".join(known_keywords),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # output is a list of strings, not a KeywordsOutput instance
    return output.keywords


# function to perform hybrid search combining semantic search and keyword matching
def hybrid_search_with_query_keywords(
    query, vstore, documents, keyword_list, llm, top_k=5
):

    # Step 1: Semantic search
    semantic_hits = vstore.similarity_search(query, k=top_k)

    # Step 2: Use GPT to extract keywords from the query
    matched_keywords = extract_keywords_with_gpt(query, llm, keyword_list)

    # print("Matched keywords:", matched_keywords)

    # Step 3: Filter docs whose metadata has any of those keywords
    keyword_hits = [
        doc
        for doc in documents
        if any(
            kw1 == kw2
            for kw1 in doc.metadata.get("matched_keywords", [])
            for kw2 in matched_keywords
        )
    ]

    # print("Keyword hits:", len(keyword_hits))

    # Step 4: Score keyword-matching documents by keyword rarity
    scored_docs = [
        (
            doc,
            score_doc(doc, matched_keywords),
        )  # original (unnormalized) list used for scoring
        for doc in keyword_hits
    ]

    # # print doc metadata and scores
    # for doc, score in scored_docs:
    #     print(f"Document: {doc.metadata.get('disease_name', 'Unknown')}, Score: {score}")
    #     print(f"Matched Keywords: {doc.metadata.get('matched_keywords', [])}")

    # Step 5: Rank and select top documents by score
    ranked_docs = sorted(scored_docs, key=lambda x: -x[1])
    top_docs = [doc for doc, score in ranked_docs if score > 0]
    top_3_docs = top_docs[:3]

    # Step 4: Merge by unique content
    merged = {doc.page_content: doc for doc in semantic_hits + top_3_docs}
    return list(merged.values())


# Main function to perform the IDSR check
def idsr_check(query: str, llm) -> AppState:
    """
    Perform hybrid search combining semantic search and keyword matching.

    Args:
        state (AppState): Application state containing the query.

    Returns:
        AppState: Updated state with search results.
    """
    # Perform hybrid search
    results = hybrid_search_with_query_keywords(
        query, vectorstore, tagged_documents, keywords, llm
    )

    disease_definitions = "\n\n".join(
        [
            f"{doc.metadata.get('disease_name', 'Unknown Disease')}:\n{doc.page_content}"
            for doc in results
        ]
    )

    # Prepare prompt for the LLM
    prompt = """
    You are a medical assistant reviewing a brief clinical case in Kenya to help identify which diseases the patient may plausibly have. You have access to several disease definitions.

    Your task is as follows:
    1. Carefully compare the case description to each disease definition.
    2. If a disease seems like a possible match based on the available information, list it and explain why.
    3. Only include rare diseases (e.g., eradicated or non-endemic to Kenya) if the match is extremely strong. Prioritize common and plausible conditions.
    4. If no disease clearly matches, say: "No strong match found."
    5. If some critical information is missing, you may ask clarifying questions — but only once.
    6. After asking clarifying questions, proceed with an assessment anyway based on what is already available.

    Case:
    {query}

    Diseases:
    {disease_definitions}

    Your response should follow this format:

    Likely matches:
    - Disease Name: [Likely] – Reason
    - Disease Name: [Probable] – Reason
    (Only include diseases that clearly fit based on the information.)

    If none:
    - No strong match found.

    Clarifying questions (optional, only if needed):
    - Question 1
    - Question 2

    At the end, always give a brief recommendation like:
    - Recommendation: "Suggest monitoring for the listed conditions." OR "No disease meets criteria based on current data — suggest gathering additional history on [x, y, z]."
    """.format(
        query=query, disease_definitions=disease_definitions
    )

    # Call the LLM to generate the answer, passing the case description and disease definitions
    llm_response = llm.invoke(prompt)
    answer_text = (
        llm_response.content.strip()
        if llm_response
        else "No relevant disease information found."
    )

    return {"answer": answer_text, "last_tool": "idsr_check"}
