from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from google.colab import userdata
import tempfile
import os

DATA_PATH = "data/"
CHROMA_PATH = "chroma"
EMBED_MODEL = "nvidia/nv-embed-v1"

nvidia_api_key = userdata.get('NVIDIA_API_KEY')
if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

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
    # Use a temporary directory for the Chroma database
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_temp_path = os.path.join(temp_dir, CHROMA_PATH)
        os.makedirs(chroma_temp_path, exist_ok=True)

        embeddings = NVIDIAEmbeddings(model=EMBED_MODEL)
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=chroma_temp_path)
        db.persist()
        print(f"Created vector DB in temporary directory '{chroma_temp_path}'")

if __name__ == "__main__":
    documents = load_pdfs(DATA_PATH)
    chunks = create_chunks(documents)
    create_vector_db(chunks)
    print("✅ Vector database built successfully!")