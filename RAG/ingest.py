from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

# Load documents

docs = []

docs.extend(
    TextLoader(
        "data/mapping_KB.txt",
        encoding="utf-8"
    ).load()
)

docs.extend(
    TextLoader(
        "data/rca_KB.txt",
        encoding="utf-8"
    ).load()
)

# Split

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)

print("Knowledge Base Indexed Successfully")