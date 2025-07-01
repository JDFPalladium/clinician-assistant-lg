import sqlite3
import pandas as pd

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")

from .state_types import AppState

# Define the SQL query tool
def sql_chain(state: AppState) -> AppState:
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
    pk_hash = state.get("pk_hash")
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
        
        def safe(val):
            if pd.isnull(val) or val in ("", "NULL"):
                return 'missing'
            return val

        summaries = []
        for _, row in df.sort_values("VisitDate", ascending=False).head(5).iterrows():
            summaries.append(f"- {row['VisitDate']}: WHO Stage {safe(row['WHOStage'])}, Weight {safe(row['Weight'])}kg, "
                             f"NextAppointmentDate {safe(row['NextAppointmentDate'])}, VisityType {safe(row['VisitType'])}, "
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
        
        def safe(val):
            if pd.isnull(val) or val in ("", "NULL"):
                return 'missing'
            return val
    
        summaries = []
        for _, row in df.sort_values("DispenseDate", ascending=False).head(5).iterrows():
            summaries.append(f"- {row['DispenseDate']}: ExpectedReturn {safe(row['ExpectedReturn'])}, Drug {safe(row['Drug'])}, "
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
        
        def safe(val):
            if pd.isnull(val) or val in ("", "NULL"):
                return 'missing'
            return val
    
        summaries = []
        for _, row in df.sort_values("OrderedbyDate", ascending=False).head(5).iterrows():
            summaries.append(f"- {row['OrderedbyDate']}: TestName {safe(row['TestName'])}, TestResult {safe(row['TestResult'])},"
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
        
        def safe(val):
            if pd.isnull(val) or val in ("", "NULL"):
                return 'missing'
            return val
        
        row = df.iloc[0]
        summary = (
            f"Sex: {safe(row['Sex'].values[0])}\n"
            f"MaritalStatus: {safe(row['MaritalStatus'].values[0])}\n"
            f"EducationLevel: {safe(row['EducationLevel'].values[0])}\n"
            f"Occupation: {safe(row['Occupation'].values[0])}\n"
            f"OnIPT: {safe(row['OnIPT'].values[0])}\n"
            f"ARTOutcomeDescription: {safe(row['ARTOutcomeDescription'].values[0])}\n"
            f"StartARTDate: {safe(row['StartARTDate'].values[0])}\n"
            f"Date Of Birth: {safe(row['DOB'].values[0])}"
        )
        return summary
    
    demographic_summary = summarize_demographics(demographic_data)
    print(demographic_summary)

    # cursor.execute("SELECT * FROM data_dictionary")
    # rows = cursor.fetchall()
    # data_dictionary = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    conn.close()

    prompt = (
        "Given the following user question, contextual clinical guidance, "
        "patient clinical data, patient lab data, patient pharmacy data, "
        "patient demographic data, answer the user question. "
        "Try to answer based on the provided data."
        "If there is essential patient information missing without which you cannot answer, "
        "do not provide an answer and instead explain what information is missing. \n\n"
        f'Question: {state["question"]}\n'
        f'Context: {state.get("rag_result", "No guidelines provided.")}\n'
        f'Patient Clinical Visits: {visits_summary}\n'
        f'Patient Pharmacy Data: {pharmacy_summary}\n'
        f'Patient Lab Data: {lab_summary}\n'
        f'Patient Demographic Data: {demographic_summary}\n'
        # f'Data Dictionary: {data_dictionary}\n'
    )
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state