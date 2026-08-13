from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

query = """
severity_type 5
resource_type 8
event_type 11
feature 68
volume 250
"""

docs = retriever.invoke(query)

for i,d in enumerate(docs,1):
    print("="*50)
    print(f"Document {i}")
    print(d.page_content[:1000])