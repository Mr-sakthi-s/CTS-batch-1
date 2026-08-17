from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = r"E:\github\CTS-batch-1\vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\Users\sadik\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

knowledge = Chroma(
    collection_name="telecom_knowledge",
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

patterns = Chroma(
    collection_name="telecom_patterns",
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

print("Knowledge:", knowledge._collection.count())
print("Patterns:", patterns._collection.count())