from .state_types import AppState
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rapidfuzz import fuzz
from langchain_core.documents import Document
from langchain_core.tools import tool
import json

llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")

# load keywords from file
with open("./guidance_docs/idsr_keywords.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]

# load vectorstore
vectorstore = FAISS.load_local("./guidance_docs/disease_vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# load tagged documents from JSON for keyword matching to document metadata
with open("./guidance_docs/tagged_documents.json", "r", encoding="utf-8") as f:
    doc_dicts = json.load(f)

tagged_documents = [Document(**d) for d in doc_dicts]

## Define helper functions

# function to find keywords in a prompt
def find_keywords_in_prompt(prompt, keywords, threshold=80):
    """
    Returns all keywords that appear in the prompt using fuzzy matching.
    
    Args:
        prompt (str): The user prompt.
        keywords (list): List of keywords to match.
        threshold (int): Fuzzy match threshold (0-100).
        
    Returns:
        list: Matched keywords.
    """
    prompt_lower = prompt.lower()
    matched = []
    for kw in keywords:
        kw_lower = kw.lower()
        # Use partial_ratio for substring-like matching
        if fuzz.partial_ratio(kw_lower, prompt_lower) >= threshold:
            matched.append(kw)
    return matched

# function to perform hybrid search combining semantic search and keyword matching
def hybrid_search_with_query_keywords(query, vectorstore, documents, keyword_list, top_k=5, match_threshold=80):
    # Step 1: Semantic search
    semantic_hits = vectorstore.similarity_search(query, k=top_k)

    # Step 2: Fuzzy match query → keywords
    matched_keywords = find_keywords_in_prompt(query, keyword_list, threshold=match_threshold)
    print(f"Matched keywords: {matched_keywords}")
    # Step 3: Filter docs whose metadata has any of those keywords
    keyword_hits = [
        doc for doc in documents
        if any(kw in doc.metadata.get("matched_keywords", []) for kw in matched_keywords)
    ]

    # print the metadata of documents hit by keyword_hits
    for doc in keyword_hits:
        print(f"Keyword hit document: {doc.metadata.get('disease_name', 'Unknown Disease')} - {doc.page_content[:100]}...")

    # Step 4: Merge by unique content
    merged = {doc.page_content: doc for doc in semantic_hits + keyword_hits}
    return list(merged.values())

# Main function to perform the IDSR check
@tool
def idsr_check(state: AppState) -> AppState:
    """
    Perform hybrid search combining semantic search and keyword matching.
    
    Args:
        state (AppState): Application state containing the query.
        
    Returns:
        AppState: Updated state with search results.
    """
    query = state.get("question", "")

    # Perform hybrid search
    results = hybrid_search_with_query_keywords(query, vectorstore, tagged_documents, keywords)

    # Now let's query the LLM to identify if any of these results are relevant to the patient condition described in the query
    if not results:
        state['answer'] = "No relevant disease information found."
        return state
    
    # Prepare prompt for the LLM
    prompt = """
    You are a medical assistant reviewing a brief clinical case to help identify which diseases the patient may plausibly have. You have access to several disease definitions.

    Your task is as follows:
    1. Carefully compare the case description to each disease definition.
    2. If a disease seems like a possible match based on the available information, list it and explain why.
    3. If no disease clearly matches, say: "No strong match found."
    4. If some critical information is missing, you may ask clarifying questions — but only once.
    5. After asking clarifying questions, proceed with an assessment anyway based on what is already available.

    Respond in this format:

    Case:
    {case_description}

    Diseases:
    {disease_definitions}

    Your response should follow this format:

    Likely matches:
    - Disease Name: [Likely] – Reason
    - Disease Name: [Likely] – Reason
    (Only include diseases that clearly fit based on the information.)

    If none:
    - No strong match found.

    Clarifying questions (optional, only if needed):
    - Question 1
    - Question 2

    At the end, always give a brief recommendation like:
    - Recommendation: "Suggest monitoring for the listed conditions." OR "No disease meets criteria based on current data — suggest gathering additional history on [x, y, z]."
    """


    # Call the LLM to generate the answer, passing the case description and disease definitions
    case_description = query
    disease_definitions = "\n\n".join([f"{doc.metadata.get('disease_name', 'Unknown Disease')}:\n{doc.page_content}" for doc in results])   
    prompt = prompt.format(case_description=case_description, disease_definitions=disease_definitions)
    llm_response = llm.invoke(prompt)   

    # Update state with results
    # state['results'] = results
    state['answer'] = llm_response.content.strip() if llm_response else "No relevant disease information found."
    return state