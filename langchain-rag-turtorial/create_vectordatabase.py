from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

from dotenv import load_dotenv
import os
import shutil
import logging
from typing import List
import argparse

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY not set in environment or .env file")
openai.api_key = api_key

CHROMA_PATH = "chroma"
DATA_PATH = "data/"  # can be a directory or a single .docx file

def load_documents(data_path: str) -> List[Document]:
    # Support single .docx file
    if os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext != ".docx":
            raise ValueError(f"Unsupported file type: {ext}. Only .docx is supported for single-file input.")
        loader = Docx2txtLoader(data_path)
        documents = loader.load()
        return documents

    # Support directory containing .docx files
    if os.path.isdir(data_path):
        loader = DirectoryLoader(data_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
        documents = loader.load()
        return documents

    raise FileNotFoundError(f"Path not found: {data_path}")

def split_text(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
    # Split documents into smaller chunks (larger defaults suit .docx paragraphs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info("Split %d documents into %d chunks.", len(documents), len(chunks))

    if not chunks:
        logging.warning("No chunks produced from documents")
        return []

    # Preview the first chunk for debugging
    first = chunks[0]
    logging.debug("First chunk preview: %s", first.page_content[:200])
    logging.debug("First chunk metadata: %s", first.metadata)

    return chunks

def save_to_chroma(chunks: List[Document], chroma_path: str = CHROMA_PATH):
    if not chunks:
        logging.warning("No chunks to save to Chroma.")
        return
    
    # Clear out the database first.
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Create a new DB from the documents.
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=chroma_path
    )
    db.persist()
    logging.info("Saved %d chunks to %s.", len(chunks), chroma_path)

def generate_data_store(data_path: str = DATA_PATH, chroma_path: str = CHROMA_PATH, chunk_size: int = 800, chunk_overlap: int = 200):
    documents = load_documents(data_path)
    if not documents:
        logging.warning("No documents found at %s", data_path)
        return
    chunks = split_text(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_to_chroma(chunks, chroma_path)

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Create a Chroma vector DB from a .docx file or a directory of .docx files.")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to a .docx file or a directory containing .docx files.")
    parser.add_argument("--chroma-path", default=CHROMA_PATH, help="Directory to persist the Chroma DB.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    generate_data_store(
        data_path=args.data_path,
        chroma_path=args.chroma_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

if __name__ == "__main__":
    main()