import os
from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# Set up OpenAI API Key
openai.api_key = os.environ['OPENAI_API_KEY']
#langchain.api_key = os.environ['LANGCHAIN_API_KEY']
# os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')     # use this line in Colab


# Define the SQLite database path
db_name = "HR_data.db"  # The database file you created
db_path = f"sqlite:///{db_name}"      # SQLite connection URI format

# Connect to the SQLite database using SQLDatabase from LangChain
db = SQLDatabase.from_uri(
    db_path,
    sample_rows_in_table_info=1,  # Adjust sample rows per table if needed
    include_tables=['employee_data_attrition'],  # Specify tables to include
    custom_table_info={'employee_data_attrition': "employee_data_attrition"}
)

# Print database dialect and table information
print("Database Dialect:", db.dialect)
print("Usable Table Names:", db.get_usable_table_names())
print("Table Information:", db.table_info)
# check SQL run 
db.run("SELECT * FROM employee_data_attrition LIMIT 10;")
# How many employees have worked here greater than 5 years?
db.run("SELECT COUNT(EmpID) FROM employee_data_attrition WHERE YearsAtCompany > 5;")

'''
Convert Question to SQL query. 
This part is to convert an user's question to SQL script
'''
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
generate_query = create_sql_query_chain(llm, db)
query = generate_query.invoke({"question": "How many employees have worked here greater than 5 years?"})

print(query)

# SELECT COUNT("EmployeeNumber") FROM employee_data_attrition WHERE "YearsAtCompany" > 5