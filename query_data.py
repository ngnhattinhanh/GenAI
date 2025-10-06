from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import argparse

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the provided context to answer the question.
Context:
{context}

Question:
{question}
"""

def main():
    # Create CLI
    parser = argparse.ArgumentParser(description="Query the vector database.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the database
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the database
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:  # Assuming relevance score threshold
        print("No relevant documents found.")
        return

    # Build context from top documents
    context_text = "\n\n".join(doc.page_content for doc, _ in results)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_prompt(question=query_text, context=context_text)
    print(prompt.to_string())

    # Prepare the LLM
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke(prompt.to_messages())

    source_docs = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"Response: {response}\n\nSources: {', '.join(source_docs)}"
    print(formatted_response)
    print("\n\n")

if __name__ == "__main__":
    main()
    