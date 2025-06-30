from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated


from .state_types import AppState, SqlChainOutputModel

db = SQLDatabase.from_uri("sqlite:///data/patient_demonstration.sqlite")
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")

# from langchain_ollama.chat_models import ChatOllama
# local_llm = ChatOllama(model="mistral:latest", temperature=0)

# setup template for sql query tool
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. For context, the database contains information about
patients' demographics, their clinical visits, and their lab tests and their pharmacy pickups.
Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. Filter PatientPKHash column using exactly the provided value: {pk_hash} if the value is provided
to get information about the patient with whom the clinician is meeting. 

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

When checking if a patient was late for an appointment, for each visit, compare the NextAppointmentDate from the previous visit to the VisitDate of the current visit.
Do not compare NextAppointmentDate to the VisitDate in the same row. Use SQL to find, for each patient, the next VisitDate after a given VisitDate, and compare it to the NextAppointmentDate from the previous visit.

Here is an example of how to do this in SQL:
SELECT
v1.PatientPKHash,
v1.VisitDate AS PreviousVisitDate,
v1.NextAppointmentDate,
v2.VisitDate AS NextVisitDate,
CASE
    WHEN v2.VisitDate <= v1.NextAppointmentDate THEN 'On time'
    ELSE 'Late'
END AS AttendanceStatus
FROM clinical_visits v1
JOIN clinical_visits v2
ON v1.PatientPKHash = v2.PatientPKHash
AND v2.VisitDate > v1.VisitDate
WHERE NOT EXISTS (
SELECT 1 FROM clinical_visits v3
WHERE v3.PatientPKHash = v1.PatientPKHash
    AND v3.VisitDate > v1.VisitDate
    AND v3.VisitDate < v2.VisitDate
)
ORDER BY v1.PatientPKHash, v1.VisitDate;

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state:AppState) -> AppState:
    """Generate SQL query to fetch information."""
    conv = state["conversation"]
    query_data = state["query_data"]

    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": conv["question"],
            # "guidelines": conv.get("rag_result", "No guidelines provided."),
            "pk_hash": conv.get("pk_hash", "")
            # "input": state["question"],
            # "pk_hash": state["pk_hash"]
        }
    )

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    query_data["query"] = result["query"]
    state["query_data"] = query_data
    return state
    # return {**state, "query": result["query"]}

def execute_query(state:AppState) -> AppState:
    """Execute SQL query."""
    query_data = state["query_data"]
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    query_data["result"] = execute_query_tool.invoke(query_data["query"])
    state["query_data"] = query_data
    return state
    # return {**state, "result": execute_query_tool.invoke(state["query"])}

def generate_answer(state:AppState) -> AppState:
    """
    Answer question using retrieved information as context.
    For awareness, NextAppointmentDate is set during the VisitDate of the same entry.
    To determine if the patient came on time to their next appointment, compare NextAppointmentDate
    with the next recorded VisitDate. For example, if a patient has a VisitDate of
    2023-01-01 and a NextAppointmentDate of 2023-01-15, check if the next VisitDate is on or before
    2023-01-15 to determine if the patient came on time.
    
    """
    conv = state["conversation"]
    query_data = state["query_data"]

    prompt = (
        "Given the following user question, context information, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {conv["question"]}\n'
        # f'Context: {conv.get("rag_result", "No guidelines provided.")}\n'
        f'SQL Query: {query_data["query"]}\n'
        f'SQL Result: {query_data["result"]}'  
        # f'Question: {state["question"]}\n'
        # f'SQL Query: {state["query"]}\n'
        # f'SQL Result: {state["result"]}'        
    )
    response = llm.invoke(prompt)
    conv["answer"] = response.content
    state["conversation"] = conv
    return state
    # return {**state, "answer": response.content}

# now define a stateful tool that does the same thing
@tool 
def sql_chain(state:AppState) -> dict:
    """
    Annotated function that takes a question string seeking information on patient data
    from a SQL database, writes an SQL query to retrieve relevant data, executes the query,
    and generates a natural language answer based on the query results.
    Returns the final answer as a string.
    """
    state = write_query(state)
    state = execute_query(state)
    print(state['query_data']['query'])
    print(state['query_data']['result'])
    state = generate_answer(state)
    print(state['conversation']['answer'])

    # Prepare output dict with only messages and conversation
    output = {
        "messages": state["messages"],
        "conversation": state["conversation"],
    }

    return output

    # return state
