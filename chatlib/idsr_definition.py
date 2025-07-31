from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.documents import Document
import json

with open("./data/processed/tagged_documents.json", "r", encoding="utf-8") as f:
    doc_dicts = json.load(f)

tagged_documents = [Document(**d) for d in doc_dicts]

class DiseaseSelectionOutput(BaseModel):
    disease_name: Optional[str] = Field(
        description="The most likely disease the user is asking about, or null if no match is confident"
    )


def select_disease_from_query(query: str, llm, tagged_docs: list[Document]) -> Optional[str]:
    disease_names = [doc.metadata.get("disease_name") for doc in tagged_docs]
    disease_list = "\n".join(f"- {name}" for name in disease_names)

    parser = PydanticOutputParser(pydantic_object=DiseaseSelectionOutput)

    prompt = ChatPromptTemplate.from_template(
        """
You are helping a clinician retrieve a disease definition from a list of IDSR diseases.

Given the following query:
"{query}"

Select the single disease from the list below that the query most likely refers to.

List of available diseases:
{disease_list}

If no match is clearly appropriate, set "disease_name" to null.

{format_instructions}
"""
    )

    chain = prompt | llm | parser
    output = chain.invoke({
        "query": query,
        "disease_list": disease_list,
        "format_instructions": parser.get_format_instructions()
    })

    return output.disease_name

def idsr_define(query: str, llm) -> dict:
    disease_name = select_disease_from_query(query, llm, tagged_documents)

    if not disease_name:
        return {
            "answer": "Sorry, I couldn't find a clear match for that disease. Please rephrase or try a different name."
        }

    # Search for matching doc
    for doc in tagged_documents:
        if doc.metadata.get("disease_name") == disease_name:
            definition = doc.page_content.strip()

            # Use LLM to generate a helpful answer
            prompt = f"""
    You are a medical assistant helping a clinician understand disease case definitions.

    Here is a user query:
    "{query}"

    Here is the official case definition for the relevant disease:
    "{definition}"

    Based on the case definition, answer the user query clearly and concisely. Do not speculate beyond the information provided.
    """
            llm_response = llm.invoke(prompt)

            return {
                "answer": llm_response.content.strip()
            }

    return {
        "answer": f"Sorry, no case definition was found for the selected disease."
    }
