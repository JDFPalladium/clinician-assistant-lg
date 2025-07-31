from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated


from .state_types import AppState

db = SQLDatabase.from_uri("sqlite:///data/patient_demonstration.sqlite")
llm = ChatOpenAI(temperature=0.0, model="gpt-4o")


system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. The database contains the following tables and columns:

table: clinical_visits
columns: PatientPKHash, VisitDate, VisitType, VisitBy, NextAppointmentDate, TCAReason, Pregnant, Breastfeeding,
    StabilityAssessment, DifferentiatedCare, WHOStage, WHOStagingOI, Height, Weight,
    EMR, Project, Adherence, AdherenceCategory, BP, OI, OIDate, CurrentRegimen, AppointmentReminderWillingness

table: lab
columns: PatientPKHash, SiteCode, OrderedByDate, TestName, TestResult 

table: pharmacy
columns: PatientPKHash, SiteCode, DispenseDate, Drug, ExpectedReturn, Duration, TreatmentType,
    RegimenLine, RegimenChangedSwitched, RegimenChangeSwitchedReason

table: demographics
columns: PatientPKHash, MFLCode, FacilityName, County, SubCounty, PartnerName, AgencyName, Sex,
    MaritalStatus, EducationLevel, Occupation, OnIPT, AgeGroup, ARTOutcomeDescription, AsOfDate, LoadDate, StartARTDate, DOB

To understand what each column means, refer to the following data dictionary: {table_info}.    

Filter PatientPKHash column using exactly the provided value: {pk_hash} if the value is provided
to get information about the patient with whom the clinician is meeting. 

If provided, create the query based on the following authoriative context from HIV clinical guidelines:
{guidelines}.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question. Use LIMIT 10 unless otherwise specified.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table. Do not join or select from any tables not listed above. 
Do not select columns not listed in the schema.

When looking for a patient's regimen information, use CurrentRegimen from clinical_visits table to see
what regimen they are on and join with the pharmacy table for other regimen details, including regimen line
and regimen switch.

Here's an example of how to do this in SQL:
SELECT r.CurrentRegimen, p.RegimenChangedSwitched, p.RegimenChangeSwitchedReason
FROM pharmacy p
JOIN regimen r ON p.PatientPKHash = r.PatientPKHash
WHERE p.PatientPKHash = '{pk_hash}'
ORDER BY p.DispenseDate DESC
LIMIT 10;


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
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: AppState) -> AppState:
    """Generate SQL query to fetch information."""

    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "table_info": db.run("SELECT * FROM data_dictionary;"),
            "input": state["question"],
            "guidelines": state.get("rag_result", "No guidelines provided."),
            "pk_hash": state.get("pk_hash", ""),
        }
    )

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    state["query"] = result["query"]  # type: ignore
    return state


def execute_query(state: AppState) -> AppState:
    """Execute SQL query."""

    execute_query_tool = QuerySQLDatabaseTool(db=db)
    state["result"] = execute_query_tool.invoke(state["query"])  # type: ignore
    return state


def generate_answer(state: AppState) -> AppState:
    """
    Answer question using retrieved information as context.
    For awareness, NextAppointmentDate is set during the VisitDate of the same entry.
    To determine if the patient came on time to their next appointment, compare NextAppointmentDate
    with the next recorded VisitDate. For example, if a patient has a VisitDate of
    2023-01-01 and a NextAppointmentDate of 2023-01-15, check if the next VisitDate is on or before
    2023-01-15 to determine if the patient came on time.

    """

    prompt = (
        "Given the following user question, context information, corresponding SQL query, "
        "and SQL result, answer the user question. If the SQL result is empty, then the SQL query was not able to retrieve any information. "
        "In that case, ignore the SQL query too and generate an answer based only on the context. \n\n"
        f'Question: {state["question"]}\n'
        f'Context: {state.get("rag_result", "No guidelines provided.")}\n'
        f'SQL Query: {state["query"]}\n'  # type: ignore
        f'SQL Result: {state["result"]}'  # type: ignore
    )
    response = llm.invoke(prompt)
    state["answer"] = response.content  # type: ignore
    return state


@tool
def sql_chain(state: AppState) -> dict:
    """
    Annotated function that takes a question string seeking information on patient data
    from a SQL database, writes an SQL query to retrieve relevant data, executes the query,
    and generates a natural language answer based on the query results.
    Returns the final answer as a string.
    """
    state = write_query(state)
    state = execute_query(state)
    state = generate_answer(state)

    return state  # type: ignore
