import re
from typing import List
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

class MultiQueryFixed:
    def __init__(self, question: str, db: SQLDatabase, llm):
        self.question = question
        self.db = db
        self.llm = llm

    # --- generate n paraphrases (multi-questions)
    def gen_multi_questions(self):
        def is_match_question(question):
            pattern = r'^\d+\.\s+'
            return re.match(pattern, question)

        def filter_valid_question(questions):
            return [q for q in questions if q and is_match_question(q)]

        template = """You are an AI language model assistant. 
        Generate five different versions of the given user question to retrieve relevant information.
        Provide only the list below format:
        1. question 1
        2. question 2
        ...
        Original question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(lambda x: x.split("\n"))
            | RunnableLambda(filter_valid_question)
            | RunnableLambda(lambda lst: [x.strip() for x in lst])
        )
        return chain

    # --- generate SQL & execute ---
    def sql_exec_chain(self):
        write_query = create_sql_query_chain(self.llm, self.db)
        exec_tool = QuerySQLDataBaseTool(db=self.db)

        def gen_sql_and_run(questions):
            results = []
            for q in questions:
                sql = write_query.invoke({"question": q})
                sql_clean = (
                    sql.replace("SQLQuery:", "")
                    .replace("```sql", "")
                    .replace("```", "")
                    .strip()
                )
                try:
                    data = exec_tool.invoke(sql_clean)
                except Exception as e:
                    data = f"Error: {e}"
                results.append({"question": q, "sql": sql_clean, "result": data})
            return results

        return RunnableLambda(gen_sql_and_run)

    # --- build formatted text for summarization ---
    @staticmethod
    def build_prompt_input(results):
        formatted = "\n\n".join(
            f"[Question {i+1}]\n{r['question']}\nSQL Query:\n{r['sql']}\nResult:\n{r['result']}"
            for i, r in enumerate(results)
        )
        return formatted

    # --- full summarization pipeline ---
    def summarize_chain(self):
        """Combine all subchains: multi-question → sql → summarize"""
        # Define summarization template
        template = """
        You are an AI data analyst. Based on the SQL queries and their results,
        summarize a single final answer that combines all insights.
        Explain briefly and logically.

        --- DATA ---
        {data}
        --- END DATA ---

        Final Answer:"""
        prompt = PromptTemplate.from_template(template)

        # Combine everything into one chain
        chain = (
            RunnablePassthrough.assign(multi_questions=self.gen_multi_questions())
            | RunnablePassthrough.assign(sql_results=lambda x: self.sql_exec_chain().invoke(x["multi_questions"]))
            | RunnablePassthrough.assign(data=lambda x: self.build_prompt_input(x["sql_results"]))
            | RunnablePassthrough.assign(answer=(prompt | self.llm | StrOutputParser()))
            | RunnableLambda(lambda x: {
                "multi_questions": x["multi_questions"],
                "sql_results": x["sql_results"],
                "final_answer": x["answer"]
            })
        )

        return chain

    # --- run everything (entry point) ---
    def run(self):
        chain = self.summarize_chain()
        return chain.invoke({"question": self.question})
