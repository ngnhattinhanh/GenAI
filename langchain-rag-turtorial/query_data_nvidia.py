import argparse
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "/tmp/tmp2xqig2ug/chroma"
EMBED_MODEL = "nvidia/nv-embed-v1"
LLM_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the provided context to answer the question.
Context:
{context}

Question:
{question}

Answer:
"""
def main():
    parser = argparse.ArgumentParser(description="Query the vector database.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    question = args.query_text

    # Load vector DB
    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search top 3 chunks
    results = db.similarity_search_with_relevance_scores(question, k=3)
    if not results:
        print("⚠️ No relevant context found.")
        exit()

    context = "\n\n".join(doc.page_content for doc, _ in results)

    # Build prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format_prompt(
        context=context, question=question
    ).to_string()

    # Query NVIDIA LLM
    llm = ChatNVIDIA(model=LLM_MODEL)
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content if hasattr(response, "content") else response)

    print("\nSources:")
    for doc, score in results:
        print("-", doc.metadata.get("source", "unknown"))

if __name__ == "__main__":
    main()
