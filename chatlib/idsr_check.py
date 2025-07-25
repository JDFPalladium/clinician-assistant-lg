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
import sqlite3

# import os


with open("./data/processed/idsr_keywords.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]

vectorstore = FAISS.load_local(
    "./data/processed/disease_vectorstore",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)


with open("./data/processed/tagged_documents.json", "r", encoding="utf-8") as f:
    doc_dicts = json.load(f)

tagged_documents = [Document(**d) for d in doc_dicts]


keyword_doc_counts = Counter()
total_docs = len(tagged_documents)

for doc in tagged_documents:
    seen = set(doc.metadata.get("matched_keywords", []))
    for kw in seen:
        keyword_doc_counts[kw] += 1

keyword_weights = {
    kw: math.log(total_docs / (1 + count)) for kw, count in keyword_doc_counts.items()
}


def score_doc(doc_to_score, matched_keywords):
    doc_keywords = set(doc_to_score.metadata.get("matched_keywords", []))
    overlap = doc_keywords & set(matched_keywords)
    return sum(keyword_weights.get(kw, 0) for kw in overlap)


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

    chain = prompt | llm | parser

    output = chain.invoke(
        {
            "query": query,
            "keyword_list": ", ".join(known_keywords),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return output.keywords


def hybrid_search_with_query_keywords(
    query, vstore, documents, keyword_list, llm, top_k=3
):

    semantic_hits = vstore.similarity_search(query, k=top_k)

    matched_keywords = extract_keywords_with_gpt(query, llm, keyword_list)

    keyword_hits = [
        doc
        for doc in documents
        if any(
            kw1 == kw2
            for kw1 in doc.metadata.get("matched_keywords", [])
            for kw2 in matched_keywords
        )
    ]

    scored_docs = [
        (
            doc,
            score_doc(doc, matched_keywords),
        )
        for doc in keyword_hits
    ]

    ranked_docs = sorted(scored_docs, key=lambda x: -x[1])
    top_docs = [doc for doc, score in ranked_docs if score > 0]
    top_5_docs = top_docs[:5]

    merged = {doc.page_content: doc for doc in semantic_hits + top_5_docs}
    return list(merged.values())


def idsr_check(query: str, llm, sitecode) -> AppState:
    """
    Perform hybrid search combining semantic search and keyword matching.

    Args:
        state (AppState): Application state containing the query.

    Returns:
        AppState: Updated state with search results.
    """

    results = hybrid_search_with_query_keywords(
        query, vectorstore, tagged_documents, keywords, llm
    )

    ## prepare to get location data
    # first, get sitecode from environment variable
    # sitecode = os.environ.get("SITECODE")
    # next, connect to location database and get county where code = sitecode
    conn = sqlite3.connect("data/processed/location_data.sqlite")
    county_cursor = conn.cursor()
    county_cursor.execute(
        "SELECT County FROM sitecode_county_xwalk WHERE Code = ?", (sitecode,)
    )
    county = county_cursor.fetchone()

    # set up connection to location database and get EpidemicInfo for any diseases in the disease_name metadata field of the results from the hybrid search
    cursor_epi = conn.cursor()
    disease_names = [doc.metadata.get("disease_name") for doc in results]
    placeholders = ",".join("?" * len(disease_names))
    query_str = f"SELECT Disease, EpidemicInfo FROM who_bulletin WHERE Disease IN ({placeholders})"
    cursor_epi.execute(query_str, disease_names)
    epidemic_info = cursor_epi.fetchall()

    # print(doc.metadata.get("disease_name") for doc in results)

    # set up connection to location database and county-specific disease prevalence and seasonality information for any diseases in the disease_name metadata field of the results from the hybrid search
    cursor_disease = conn.cursor()
    if county:  # Ensure county is not None
        county_name = county[0]
        disease_names = [doc.metadata.get("disease_name") for doc in results]
        placeholders = ",".join("?" * len(disease_names))
        query_str = f"SELECT County, Disease, Prevalence, Notes FROM county_disease_info WHERE County = ? AND Disease IN ({placeholders})"
        cursor_disease.execute(query_str, (county_name, *disease_names))
        county_info = cursor_disease.fetchall()

        # Get climate information for the county from the rainy seasons table
        # Get the current month
        from datetime import datetime

        current_month = datetime.now().strftime("%B")  # Full month name, e.g. "March"
        cursor_disease.execute(
            "SELECT RainySeason FROM county_rainy_seasons WHERE County = ? and Month = ?",
            (county_name, current_month),
        )
        rainy_season = cursor_disease.fetchone()
        rainy_season = rainy_season[0] if rainy_season else "Unknown"

    # close the connection
    conn.close()

    disease_definitions = "\n\n".join(
        [
            f"### Disease: {doc.metadata.get('disease_name', 'Unknown Disease')}:\n{doc.page_content}"
            for doc in results
        ]
    )

    prompt = """
    You are a medical assistant reviewing a brief clinical case in Kenya to help identify which diseases the patient may plausibly have. 
    You have access to several disease definitions. You also have access to information about the prevalence of each disease in the county
    where the patient is located. The prevalence of some diseases varies by season, and some diseases are also more likely when there is a
    declared epidemic. Information on the timing of the rainy season and any declared epidemics is also provided.

    ## Instructions:
    1. Carefully compare the case description to each disease definition, taking into account the prevalence and seasonality information.
    2. If a disease seems like a possible match based on the available information, list it and explain why.
    3. Only include rare diseases, or diseases that don't fit seasonally, if the match is extremely strong. Prioritize common and plausible conditions.
    4. Only list diseases if there are plausible matches based on the case and context. If no plausible matches are found, do not list any diseases.
    5. If the information provided is insufficient or ambiguous, prioritize asking clarifying questions before making any recommendations.
    6. Clarifying questions may include inquiries about specific symptoms, patient demographics, exposures, travel history, or other relevant clinical details.
    7. Provide a brief recommendation on next steps only if confident matches are identified or after clarifications are obtained.


    ## Case:
    {query}

    ## Diseases:
    {disease_definitions}

    ## Locational context:
    In {county_name}, the current rainy season status is {rainy_season}.
    
    The above diseases have the following prevalence in the county where the patient is located:
    {county_info}

    Here are any relevant epidemic alerts for these diseases:
    {epidemic_info}

    ## Expected Output

    If applicable, list possible disease matches with explanations.

    If needed, list clarifying questions to better understand the case.

    Provide a brief recommendation on next steps if appropriate.

    If no matches or recommendations are possible, focus on clarifying questions.


    """.format(
        query=query,
        disease_definitions=disease_definitions,
        county_name=county_name if county else "Unknown County",
        rainy_season=rainy_season if county else "Unknown",
        county_info=(
            "\n".join(
                [
                    f"- {row[0]}, {row[1]}, Prevalence: {row[2]}, Notes: {row[3]}"
                    for row in county_info
                ]
            )
            if county
            else "No county information available."
        ),
        epidemic_info=(
            "\n".join([f"- {row[0]}: {row[1]}" for row in epidemic_info])
            if epidemic_info
            else "No epidemic information available."
        ),
    )
    print(prompt)
    llm_response = llm.invoke(prompt)
    answer_text = (
        llm_response.content.strip()
        if llm_response
        else "No relevant disease information found."
    )

    # Set up context to return.
    # First, use an LLM to identify which diseases from disease_definitions were mentioned in the answer_text
    disease_names_in_answer = [
        doc.metadata.get("disease_name")
        for doc in results
        if doc.metadata.get("disease_name") in answer_text
    ]
    # Next, filter the results to only include those diseases
    filtered_results = [
        doc
        for doc in results
        if doc.metadata.get("disease_name") in disease_names_in_answer
    ]
    # Finally, create context string with only those diseases, plus any county_info and epidemic_info
    context_parts = []
    if filtered_results:
        context_parts.append(
            "### Disease Definitions:\n"
            + "\n\n".join(
                [
                    f"### Disease: {doc.metadata.get('disease_name', 'Unknown Disease')}:\n{doc.page_content}"
                    for doc in filtered_results
                ]
            )
        )
    if county and county_info:
        context_parts.append(
            "### County Disease Information:\n"
            + "\n".join(
                [
                    f"- {row[0]}, {row[1]}, Prevalence: {row[2]}, Seasonality: {row[3]}"
                    for row in county_info
                ]
            )
        )
    if epidemic_info:
        context_parts.append(
            "### Epidemic Information:\n"
            + "\n".join([f"- {row[0]}: {row[1]}" for row in epidemic_info])
        )

    return {"answer": answer_text, "last_tool": "idsr_check", "context": context_parts}  # type: ignore
