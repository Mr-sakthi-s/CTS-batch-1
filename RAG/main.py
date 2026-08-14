from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ==================================================
# EMBEDDING MODEL
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\Users\sakthi murugan\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

# ==================================================
# KNOWLEDGE COLLECTION
# ==================================================

knowledge_db = Chroma(
    collection_name="telecom_knowledge",
    persist_directory="vector_db",
    embedding_function=embeddings
)

knowledge_retriever = knowledge_db.as_retriever(
    search_kwargs={"k": 3}
)

# ==================================================
# PATTERN COLLECTION
# ==================================================

pattern_db = Chroma(
    collection_name="telecom_patterns",
    persist_directory="vector_db",
    embedding_function=embeddings
)

pattern_retriever = pattern_db.as_retriever(
    search_kwargs={"k": 5}
)

# ==================================================
# LLM
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

# ==================================================
# RCA GENERATION
# ==================================================

def generate_rca(ml_output):

    query = f"""
Predicted Fault Severity: {ml_output['predicted_fault_severity']}

Severity Type: {ml_output['severity_type']}

Resource Type: {ml_output['resource_type']}

Event Types:
{', '.join(ml_output['event_types'])}

Log Features:
{', '.join(ml_output['log_features'])}

Volume:
{ml_output['volume']}
"""

    # ==========================================
    # KNOWLEDGE RETRIEVAL
    # ==========================================

    knowledge_docs = knowledge_retriever.invoke(query)

    knowledge_context = "\n\n".join(
        doc.page_content
        for doc in knowledge_docs
    )

    # ==========================================
    # PATTERN RETRIEVAL
    # ==========================================

    pattern_docs = pattern_retriever.invoke(query)

    pattern_context = "\n\n".join(
        doc.page_content
        for doc in pattern_docs
    )

    # ==========================================
    # PROMPT
    # ==========================================

    
    prompt = f"""
You are an Expert Telecom Root Cause Analysis Agent.

NETWORK INCIDENT
================

{query}

==================================================
TELECOM KNOWLEDGE BASE
==================================================

{knowledge_context}

==================================================
SIMILAR HISTORICAL INCIDENTS
==================================================

{pattern_context}

==================================================
TASK
==================================================

Use BOTH:

1. Telecom domain knowledge.
2. Similar historical incidents.

Generate a professional RCA report containing:

1. Probable Root Cause
2. Confidence Score (%)
3. Technical Explanation
4. Similar Historical Incident Analysis
5. Immediate Actions
6. Preventive Actions
7. Maintenance Recommendations
8. Risk Level

Rules:

- Base conclusions on retrieved evidence.
- Mention which historical patterns support the diagnosis.
- If historical incidents conflict with knowledge base,
  prefer knowledge base information.
- Keep recommendations practical and technical.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        answer = ""

        for block in response.content:

            if isinstance(block, dict):
                answer += block.get("text", "")

        return answer

    return response.content


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    ml_output = {
        "predicted_fault_severity": 2,
        "severity_type": "severity_type 2",
        "resource_type": "resource_type 5",
        "event_types": [
            "event_type 10",
            "event_type 12",
            "event_type 14"
        ],
        "log_features": [
            "feature 64",
            "feature 82",
            "feature 91"
        ],
        "volume": 250
    }

    result = generate_rca(ml_output)

    print("\n")
    print("=" * 100)
    print("ROOT CAUSE ANALYSIS REPORT")
    print("=" * 100)

    print(result)