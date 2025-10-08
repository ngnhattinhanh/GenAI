import argparse
from pyparsing import itemgetter

from langchain_community.utils import SQLDatabase
from langchain.chains import create_sql_database_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "chroma"
EMBED_MODEL = "nvidia/nv-embed-v1"
LLM_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

#define the SQLlite database file path
db_name = "Finance_data.db"
db_path = f"sqlite:///{db_name}"

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
def main():
    parser = argparse.ArgumentParser(description="Query the vector database.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    question = args.query_text
    #Connect to the database
    db = SQLDatabase.from_uri(
        db_path,
        include_tables=[
            "AsamaMove_Sales",
            "Supermarket_Sales",
            "Employee_Sales",
        ]
    )

    # Print database dialect and table information
    print(f"Database Dialect: {db.dialect}")
    print(f"Tables in Database: {db.get_table_names()}")
    print(f"Table Info: {db.get_table_info()}")

    #define models
    llm = ChatNVIDIA(model=LLM_MODEL)
    excute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_database_chain(llm, db)

    # Evaluate the chain with user input for {question}
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(result=itemgetter("query")) 
        | excute_query
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    # question -> write_query -> {query} -> excute_query -> {result} -> answer_prompt -> llm -> StrOutputParser -> response

    # Example usage
    response = chain.invoke({"question": question})
    print("Response: ", response)
