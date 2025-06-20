from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated

from .state_types import State
db = SQLDatabase.from_uri("sqlite:///data/patient_demonstration.sqlite")
llm = ChatOpenAI(temperature = 0.0, model="gpt-4o")

# setup template for sql query tool
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

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


def write_query(state: State) -> State:
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {**state, "query": result["query"]}

def execute_query(state: State) -> State:
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {**state, "result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State) -> State:
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}

# now define a stateful tool that does the same thing
@tool 
def sql_chain(state: State) -> State:
    """
    Annotated function that takes a question string seeking information on patient data
    from a SQL database, writes an SQL query to retrieve relevant data, executes the query,
    and generates a natural language answer based on the query results.
    Returns the final answer as a string.
    """
    state = write_query(state)
    state = execute_query(state)
    state = generate_answer(state)
    return state
