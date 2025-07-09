import sqlite3
import pandas as pd

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")

# from langchain_ollama.chat_models import ChatOllama
# local_llm = ChatOllama(model="mistral:latest", temperature=0)

from .state_types import AppState

# define helper functions
def safe(val):
    if pd.isnull(val) or val in ("", "NULL"):
        return 'missing'
    return val

# function to return only year of date
def extract_year(date_str):
    if pd.isnull(date_str) or date_str in ("", "NULL"):
        return 'missing'
    try:
        return pd.to_datetime(date_str).year
    except Exception as e:
        return 'invalid date'

# Define the SQL query tool
def sql_chain(query: str, rag_result: str, pk_hash: str) -> dict:
    """
    Annotated function that takes a patient identifer (pk_hash) and returns
    all data related to that patient from the SQL database.
    It writes an SQL query to retrieve relevant data, executes the query,
    and generates a natural language answer based on the query results.
    Returns the final answer as a string.
    The function uses the QuerySQLDatabaseTool to handle the SQL operations.
    The state should contain the following fields:
    - question: str - the question seeking information on patient data
    - pk_hash: str - the patient identifier to query the database
    - rag_result: str - context information from the guidelines retrieval
    The function will update the state with the answer to the question.
    The answer will be generated based on the SQL query results and the context information.
    The function will return the updated state with the answer.
    """

    if not pk_hash:
        raise ValueError("pk_hash is required in state for SQL queries.")

    conn = sqlite3.connect('data/patient_demonstration.sqlite')
    cursor = conn.cursor() 

    # Write the SQL query using the QuerySQLDatabaseTool
    cursor.execute("SELECT * FROM clinical_visits WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    rows = cursor.fetchall()
    visits_data = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    def summarize_visits(df):
        if df.empty:
            return "No clinical visit data available."

        summaries = []
        summaries = []
        for idx, (_, row) in enumerate(df.sort_values("VisitDate", ascending=False).head(5).iterrows(), start=1):
            summaries.append(
                f"Clinical Visit {idx} most recent, Year of visit {extract_year(row['VisitDate'])}: WHO Stage {safe(row['WHOStage'])}, Weight {safe(row['Weight'])}kg, "
                f"NextAppointmentDate {extract_year(safe(row['NextAppointmentDate']))}, VisitType {safe(row['VisitType'])}, "
                f"VisitBy {safe(row['VisitBy'])}, Pregnant {safe(row['Pregnant'])}, Breastfeeding {safe(row['Breastfeeding'])}, "
                f"WHOStage {safe(row['WHOStage'])}, StabilityAssessment {safe(row['StabilityAssessment'])}, "
                f"DifferentiatedCare {safe(row['DifferentiatedCare'])}, WHOStagingOI {safe(row['WHOStagingOI'])}, "
                f"Height {safe(row['Height'])}cm, Adherence {safe(row['Adherence'])}, BP {safe(row['BP'])}, "
                f"OI {safe(row['OI'])}, CurrentRegimen {safe(row['CurrentRegimen'])}"
            )
        return "\n".join(summaries)

    visits_summary = summarize_visits(visits_data)

    print(visits_summary)

    cursor.execute("SELECT * FROM pharmacy WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    rows = cursor.fetchall()
    pharmacy_data = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    def summarize_pharmacy(df):
        if df.empty:
            return "No pharmacy data available."
    
        summaries = []
        for idx, (_, row) in enumerate(df.sort_values("DispenseDate", ascending=False).head(5).iterrows(), start=1):
            summaries.append(f"Pharmacy Visit {idx} most recent, Year of visit {extract_year(row['DispenseDate'])}, "
                             f"ExpectedReturn {extract_year(row['ExpectedReturn'])}, Drug {safe(row['Drug'])}, "
                             f"Duration {safe(row['Duration'])}, TreatmentType {safe(row['TreatmentType'])}, "
                             f"RegimenLine {safe(row['RegimenLine'])}, "
                             f"RegimenChangedSwitched {safe(row['RegimenChangedSwitched'])}, "
                             f"RegimenChangeSwitchedReason {safe(row['RegimenChangeSwitchedReason'])}, "
            )
        return "\n".join(summaries)

    pharmacy_summary = summarize_pharmacy(pharmacy_data)

    print(pharmacy_summary)

    cursor.execute("SELECT * FROM lab WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    rows = cursor.fetchall()
    lab_data = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    def summarize_lab(df):
        if df.empty:
            return "No lab data available."
    
        summaries = []
        for idx, (_, row) in enumerate(df.sort_values("OrderedbyDate", ascending=False).head(5).iterrows(), start=1):
            summaries.append(f"Lab Test {idx} most recent, Year of test {extract_year(row['OrderedbyDate'])}. TestName {safe(row['TestName'])}, TestResult {safe(row['TestResult'])},"
            )
        return "\n".join(summaries)

    lab_summary = summarize_lab(lab_data)

    print(lab_summary)

    cursor.execute("SELECT * FROM demographics WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    rows = cursor.fetchall()
    demographic_data = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    def summarize_demographics(df):
        if df.empty:
            return "No demographic data available."
        
        def calculate_age(dob):
            if pd.isnull(dob) or dob in ("", "NULL"):
                return 'missing'
            try:
                dob = pd.to_datetime(dob)
                today = pd.to_datetime("today")
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return age
            except Exception as e:
                return 'invalid date'

        row = df.iloc[0]
        summary = (
            f"Sex: {safe(row['Sex'])}\n"
            f"MaritalStatus: {safe(row['MaritalStatus'])}\n"
            f"EducationLevel: {safe(row['EducationLevel'])}\n"
            f"Occupation: {safe(row['Occupation'])}\n"
            f"OnIPT: {safe(row['OnIPT'])}\n"
            f"ARTOutcomeDescription: {safe(row['ARTOutcomeDescription'])}\n"
            f"StartARTDate: {safe(row['StartARTDate'])}\n"
            f"Age: {calculate_age(safe(row['DOB']))}"
        )
        return summary
    
    demographic_summary = summarize_demographics(demographic_data)

    # cursor.execute("SELECT * FROM data_dictionary")
    # rows = cursor.fetchall()
    # data_dictionary = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    conn.close()

    prompt = (
        "You are a clinical assistant. Given the user question, clinical guideline context, "
        "and summarized patient data below, answer the question accurately and concisely. "
        "Only use the provided data; do not guess or hallucinate. "
        "If essential patient information is missing, explain what is missing instead of guessing. "
        "Please answer in no more than 100 words. \n\n"
        f"Question: {query}\n"
        f"Guideline Context: {rag_result}\n"
        f"Clinical Visits Summary:\n{visits_summary}\n"
        f"Pharmacy Summary:\n{pharmacy_summary}\n"
        f"Lab Summary:\n{lab_summary}\n"
        f"Demographic Summary:\n{demographic_summary}\n"
    )
    print("Prompt length (chars):", len(prompt))

    response = llm.invoke(prompt)
    answer_text = response.content
    return {
        "answer": answer_text,
        "last_tool": "sql_chain"
    }