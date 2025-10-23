import argparse
from pyparsing import itemgetter
import re

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain, LLMChain

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "chroma"
EMBED_MODEL = "nvidia/nv-embed-v1"
LLM_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

#define the SQLlite database file path
db_name = "HR_data.db"
db_path = f"sqlite:///{db_name}"

answer_prompt = PromptTemplate.from_template(
    """You are an AI SQL assistant. Given a question, SQL query, and query result, provide a detailed answer.

Question: {question}
SQL query: {query}
SQL result: {result}

Answer: """
)

def clean_sql_query(query: str) -> str:
    """Remove 'SQLQuery:' or markdown formatting."""
    if not query:
        return ""
    # Remove "SQLQuery:" prefix and markdown code fences
    query = re.sub(r"(?i)sql\s*query\s*[:\-]*", "", query)
    query = re.sub(r"```sql|```", "", query)
    query = query.strip()
    return query
    
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
    print("--------------------------------")
    print(f"Database Dialect: {db.dialect}")
    print("--------------------------------")
    print(f"Tables in Database: {db.get_table_names()}")
    print("--------------------------------")
    print(f"Table Info: {db.get_table_info()}")

    #define models
    llm = ChatNVIDIA(model=LLM_MODEL)
    print("Done LLM")
    execute_query = QuerySQLDataBaseTool(db=db)
    print("Done excute_query")
    write_query = create_sql_query_chain(llm, db, k=10)
    print("Done write_query")

    # Evaluate the chain with user input for {question}
    # --- Tạo chain chính
    def safe_execute(query: str):
        """Chỉ chạy SQL nếu hợp lệ"""
        query = clean_sql_query(query)
        if not query or len(query.split()) < 2:
            return "No valid SQL query generated."
        try:
            result = execute_query.invoke(query)
            return result
        except Exception as e:
            return f"SQL execution error: {str(e)}"

    chain = (
        {
            "question": RunnablePassthrough(),
            "query": write_query,
        }
        | RunnablePassthrough.assign(result=lambda x: safe_execute(x["query"]))
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    # question -> write_query -> {query} -> excute_query -> {result} -> answer_prompt -> llm -> StrOutputParser -> response

    # Example usage

    print("--------------------------------")
    print("🧠 Question: ", question)

    # 1️⃣ LLM sinh SQL
    sql_query = write_query.invoke({"question": question})
    print("🧱 [write_query output]:")
    print(sql_query)

    print("--------------------------------")
    # 2️⃣ Chạy SQL thật
    sql_result = safe_execute(sql_query)
    print("\n💾 [safe_execute output]:")
    print(sql_result)

    print("--------------------------------")
    # 3️⃣ Xây prompt hoàn chỉnh
    formatted_prompt = answer_prompt.format(
        question=question,
        query=sql_query,
        result=sql_result
    )
    print("\n🧩 [answer_prompt result]:")
    print(formatted_prompt)

    response = chain.invoke({"question": question})
    print("--------------------------------")
    print("✅ Response: ", response)

if __name__ == "__main__":
    main()
