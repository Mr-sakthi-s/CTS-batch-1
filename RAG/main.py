from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# =========================
# Embedding Model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Vector DB
# =========================

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# =========================
# LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

# =========================
# RAG FUNCTION
# =========================

def generate_rca(ml_output):

    query = f"""
    Predicted Fault Severity:
    {ml_output['predicted_fault_severity']}

    Severity Type:
    {ml_output['severity_type']}

    Resource Type:
    {ml_output['resource_type']}

    Event Types:
    {', '.join(ml_output['event_types'])}

    Log Features:
    {', '.join(ml_output['log_features'])}

    Volume:
    {ml_output['volume']}
    """

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are an expert Telecom Root Cause Analysis Agent.

    NETWORK INCIDENT:

    {query}

    KNOWLEDGE BASE:

    {context}

    Generate:

    1. Probable Root Cause
    2. Confidence Score
    3. Explanation
    4. Immediate Actions
    5. Preventive Actions
    6. Maintenance Recommendations
    7. Risk Level
    """

    response = llm.invoke(prompt)

    answer = ""

    if isinstance(response.content, list):

        for block in response.content:

            if isinstance(block, dict):
                answer += block.get("text", "")

    else:
        answer = response.content

    return answer


# ==========================================
# TEMPORARY TEST DATA
# REMOVE AFTER ML IS READY
# ==========================================

if __name__ == "__main__":

    ml_output = {
        "predicted_fault_severity": 2,
        "severity_type": "severity_type 5",
        "resource_type": "resource_type 8",
        "event_types": [
            "event_type 11",
            "event_type 12",
            "event_type 14"
        ],
        "log_features": [
            "feature 68",
            "feature 82",
            "feature 91"
        ],
        "volume": 250
    }

    result = generate_rca(ml_output)
    print(result)