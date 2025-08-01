import sqlite3
import pandas as pd

# import os
from .helpers import describe_relative_date


def safe(val):
    if pd.isnull(val) or val in ("", "NULL"):
        return "missing"
    return val


# define a function that takes in a date string and returns a relative date description


def extract_year(date_str):
    if pd.isnull(date_str) or date_str in ("", "NULL"):
        return "missing"
    try:
        return pd.to_datetime(date_str).year
    except (ValueError, TypeError):
        return "invalid date"


def sql_chain(query: str, llm, rag_result: str, pk_hash: str) -> dict:
    """
    Annotated function that takes a patient identifer (pk_hash) and returns
    all data related to that patient from the SQL database.
    It writes an SQL query to retrieve relevant data, executes the query,
    and generates a natural language answer based on the query results.
    Returns the final answer as a string.
    The state should contain the following fields:
    - question: str - the question seeking information on patient data
    - pk_hash: str - the patient identifier to query the database
    - rag_result: str - context information from the guidelines retrieval
    The function returns a dict with answer and last_tool keys.
    """

    # pk_hash = os.environ.get("PK_HASH")
    if not pk_hash:
        raise ValueError("pk_hash is required in state for SQL queries.")

    conn = sqlite3.connect("data/processed/patient_demonstration.sqlite")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM clinical_visits WHERE PatientPKHash = :pk_hash",
        {"pk_hash": pk_hash},
    )
    rows = cursor.fetchall()
    visits_data = pd.DataFrame(
        rows, columns=[column[0] for column in cursor.description]
    )

    def summarize_visits(df):
        if df.empty:
            return "No clinical visit data available."

        # Ensure VisitDate and NextAppointmentDate are datetime
        df = df.copy()
        df["VisitDate"] = pd.to_datetime(df["VisitDate"], errors="coerce")
        df["NextAppointmentDate"] = pd.to_datetime(
            df["NextAppointmentDate"], errors="coerce"
        )

        summaries = []
        ordinal_map = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
        for idx, (_, row) in enumerate(
            df.sort_values("VisitDate", ascending=False).head(5).iterrows(), start=1
        ):
            ordinal = ordinal_map.get(idx, f"{idx}th")
            summaries.append(
                f"{ordinal} most recent clinical visit, {describe_relative_date(row['VisitDate'])}: WHO Stage {safe(row['WHOStage'])}, Weight {safe(row['Weight'])}kg, "
                f"NextAppointmentDate {describe_relative_date(safe(row['NextAppointmentDate']))}, VisitType {safe(row['VisitType'])}, "
                f"VisitBy {safe(row['VisitBy'])}, Pregnant {safe(row['Pregnant'])}, Breastfeeding {safe(row['Breastfeeding'])}, "
                f"WHOStage {safe(row['WHOStage'])}, StabilityAssessment {safe(row['StabilityAssessment'])}, "
                f"DifferentiatedCare {safe(row['DifferentiatedCare'])}, WHOStagingOI {safe(row['WHOStagingOI'])}, "
                f"Height {safe(row['Height'])}cm, Adherence {safe(row['Adherence'])}, BP {safe(row['BP'])}, "
                f"OI {safe(row['OI'])}, CurrentRegimen {safe(row['CurrentRegimen'])}"
            )
        return "\n".join(summaries)

    visits_summary = summarize_visits(visits_data)

    cursor.execute(
        "SELECT * FROM pharmacy WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash}
    )
    rows = cursor.fetchall()
    pharmacy_data = pd.DataFrame(
        rows, columns=[column[0] for column in cursor.description]
    )

    def summarize_pharmacy(df):
        if df.empty:
            return "No pharmacy data available."

        # Ensure DispenseDate and ExpectedReturn are datetime
        df = df.copy()
        df["DispenseDate"] = pd.to_datetime(df["DispenseDate"], errors="coerce")
        df["ExpectedReturn"] = pd.to_datetime(df["ExpectedReturn"], errors="coerce")

        summaries = []
        ordinal_map = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
        for idx, (_, row) in enumerate(
            df.sort_values("DispenseDate", ascending=False).head(5).iterrows(), start=1
        ):
            ordinal = ordinal_map.get(idx, f"{idx}th")
            summaries.append(
                f"{ordinal} most recent pharmacy visit, {describe_relative_date(row['DispenseDate'])}, "
                f"ExpectedReturn {describe_relative_date(row['ExpectedReturn'])}, Drug {safe(row['Drug'])}, "
                f"Duration {safe(row['Duration'])}, TreatmentType {safe(row['TreatmentType'])}, "
                f"RegimenLine {safe(row['RegimenLine'])}, "
                f"RegimenChangedSwitched {safe(row['RegimenChangedSwitched'])}, "
                f"RegimenChangeSwitchedReason {safe(row['RegimenChangeSwitchedReason'])}, "
            )
        return "\n".join(summaries)

    pharmacy_summary = summarize_pharmacy(pharmacy_data)

    cursor.execute(
        "SELECT * FROM lab WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash}
    )
    rows = cursor.fetchall()
    lab_data = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    def summarize_lab(df):
        if df.empty:
            return "No lab data available."

        # Ensure OrderedbyDate is datetime
        df = df.copy()
        df["OrderedbyDate"] = pd.to_datetime(df["OrderedbyDate"], errors="coerce")

        summaries = []
        ordinal_map = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
        for idx, (_, row) in enumerate(
            df.sort_values("OrderedbyDate", ascending=False).head(5).iterrows(), start=1
        ):
            summaries.append(
                f"{ordinal_map.get(idx, idx)} most recent lab test, {describe_relative_date(row['OrderedbyDate'])}. TestName {safe(row['TestName'])}, TestResult {safe(row['TestResult'])},"
            )
        return "\n".join(summaries)

    lab_summary = summarize_lab(lab_data)

    cursor.execute(
        "SELECT * FROM demographics WHERE PatientPKHash = :pk_hash",
        {"pk_hash": pk_hash},
    )
    rows = cursor.fetchall()
    demographic_data = pd.DataFrame(
        rows, columns=[column[0] for column in cursor.description]
    )

    def summarize_demographics(df):
        if df.empty:
            return "No demographic data available."

        def calculate_age(dob):
            if pd.isnull(dob) or dob in ("", "NULL"):
                return "missing"
            try:
                dob = pd.to_datetime(dob)
                today = pd.to_datetime("today")
                age = (
                    today.year
                    - dob.year
                    - ((today.month, today.day) < (dob.month, dob.day))
                )
                return age
            except (ValueError, TypeError):
                return "invalid date"

        df = df.copy()
        df["StartARTDate"] = pd.to_datetime(df["StartARTDate"], errors="coerce")

        row = df.iloc[0]
        summary = (
            f"Sex: {safe(row['Sex'])}\n"
            f"MaritalStatus: {safe(row['MaritalStatus'])}\n"
            f"EducationLevel: {safe(row['EducationLevel'])}\n"
            f"Occupation: {safe(row['Occupation'])}\n"
            f"OnIPT: {safe(row['OnIPT'])}\n"
            f"ARTOutcomeDescription: {safe(row['ARTOutcomeDescription'])}\n"
            f"StartARTDate: {describe_relative_date(row['StartARTDate'])}\n"
            f"Age: {calculate_age(safe(row['DOB']))}"
        )
        return summary

    demographic_summary = summarize_demographics(demographic_data)

    conn.close()

    prompt = (
        "You are a clinical assistant. Given the user question, clinical guideline context, "
        "and summarized patient data below, answer the question accurately and concisely. "
        "Only use the provided data; do not guess or hallucinate. "
        "If essential patient information is missing, explain what is missing instead of guessing. \n\n"
        f"Question: {query}\n"
        f"Guideline Context: {rag_result}\n"
        f"Clinical Visits Summary:\n{visits_summary}\n"
        f"Pharmacy Summary:\n{pharmacy_summary}\n"
        f"Lab Summary:\n{lab_summary}\n"
        f"Demographic Summary:\n{demographic_summary}\n"
    )
    print("Prompt length (chars):", len(prompt))
    print(prompt)
    response = llm.invoke(prompt)
    answer_text = response.content
    return {"answer": answer_text, "last_tool": "sql_chain"}
