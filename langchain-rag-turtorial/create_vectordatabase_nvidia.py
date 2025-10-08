from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
import argparse
import os

CHROMA_PATH = "chroma"
EMBED_MODEL = "nvidia/nv-embed-v1"

def load_pdfs(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

def create_chunks(docs, chunk_size=800, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def create_vector_db(docs):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL)
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Created vector DB in temporary directory '{CHROMA_PATH}'")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directory containing PDFs to process.")
    parser.add_argument("data_path", type=str, help="The directory containing PDF files.")
    args = parser.parse_args()
    DATA_PATH = args.data_path
    documents = load_pdfs(DATA_PATH)
    chunks = create_chunks(documents)
    create_vector_db(chunks)
    print("✅ Vector database built successfully!")
