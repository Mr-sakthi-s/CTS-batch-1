import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ==================================================
# EMBEDDING MODEL
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# EVALUATION EMBEDDINGS
# ==================================================

eval_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# VECTOR DATABASES
# ==================================================

knowledge_db = Chroma(
    collection_name="telecom_knowledge",
    persist_directory="vector_db",
    embedding_function=embeddings
)

pattern_db = Chroma(
    collection_name="telecom_patterns",
    persist_directory="vector_db",
    embedding_function=embeddings
)

knowledge_retriever = knowledge_db.as_retriever(
    search_kwargs={"k": 3}
)

pattern_retriever = pattern_db.as_retriever(
    search_kwargs={"k": 5}
)

# ==================================================
# MODEL
# ==================================================

llm = ChatOllama(
    model="telecom-copilot",
    temperature=0
)

# ==================================================
# SIMILARITY FUNCTION
# ==================================================

def similarity(text1, text2):

    emb1 = eval_model.encode(text1)
    emb2 = eval_model.encode(text2)

    return cosine_similarity(
        [emb1],
        [emb2]
    )[0][0]

# ==================================================
# FINE-TUNED MODEL ONLY
# ==================================================

def generate_llm_only(row):

    prompt = f"""
You are a Telecom RCA Expert.

Severity:
{row['severity_category']}

Events:
{row['event_categories']}

Resources:
{row['resource_categories']}

Log Features:
{row['log_feature_groups']}

Event Count:
{row['event_count']}

Total Volume:
{row['total_log_volume']}

Generate:

1. Root Cause
2. Solution
"""

    response = llm.invoke(prompt)

    return response.content

# ==================================================
# FINE-TUNED MODEL + RAG
# ==================================================

def generate_rag(row):

    query = f"""
Severity:
{row['severity_category']}

Events:
{row['event_categories']}

Resources:
{row['resource_categories']}

Log Features:
{row['log_feature_groups']}

Event Count:
{row['event_count']}

Total Volume:
{row['total_log_volume']}
"""

    # --------------------------
    # Knowledge Retrieval
    # --------------------------

    knowledge_docs = knowledge_retriever.invoke(query)

    knowledge_context = "\n\n".join(
        doc.page_content
        for doc in knowledge_docs
    )

    # --------------------------
    # Pattern Retrieval
    # --------------------------

    pattern_docs = pattern_retriever.invoke(query)

    pattern_context = "\n\n".join(
        doc.page_content
        for doc in pattern_docs
    )

    # --------------------------
    # Prompt
    # --------------------------

    prompt = f"""
You are an Expert Telecom RCA Agent.

CURRENT INCIDENT
================

{query}

TELECOM KNOWLEDGE BASE
======================

{knowledge_context}

SIMILAR HISTORICAL INCIDENTS
============================

{pattern_context}

Generate:

1. Root Cause
2. Solution
3. Prevention
4. Confidence Score

Use both:
- Telecom Knowledge
- Historical Patterns
"""

    response = llm.invoke(prompt)

    return response.content

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(
    r"E:\github\CTS-batch-1\RAG\data\combined_data.csv"
)

# ==================================================
# RANDOM 5 INCIDENTS
# ==================================================

test_df = df.sample(
    n=5,
    random_state=42
)

# ==================================================
# EVALUATION
# ==================================================

llm_scores = []
rag_scores = []

for _, row in test_df.iterrows():

    ground_truth = f"""
Root Cause:
{row['root_cause_description']}

Solution:
{row['solution_description']}
"""

    print("\n")
    print("=" * 100)
    print(f"INCIDENT ID : {row['id']}")
    print("=" * 100)

    # ------------------------------------------------
    # Fine Tuned Model
    # ------------------------------------------------

    llm_output = generate_llm_only(row)

    llm_score = similarity(
        llm_output,
        ground_truth
    )

    # ------------------------------------------------
    # Fine Tuned + RAG
    # ------------------------------------------------

    rag_output = generate_rag(row)

    rag_score = similarity(
        rag_output,
        ground_truth
    )

    llm_scores.append(llm_score)
    rag_scores.append(rag_score)

    winner = "RAG" if rag_score > llm_score else "LLM"

    # ------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------

    print("\nGROUND TRUTH")
    print("-" * 50)
    print(ground_truth)

    print("\nLLM OUTPUT")
    print("-" * 50)
    print(llm_output)

    print("\nRAG OUTPUT")
    print("-" * 50)
    print(rag_output)

    print("\nSCORES")
    print("-" * 50)
    print(f"LLM Score : {llm_score:.4f}")
    print(f"RAG Score : {rag_score:.4f}")

    print(f"\nWinner : {winner}")

# ==================================================
# FINAL SUMMARY
# ==================================================

avg_llm = np.mean(llm_scores)
avg_rag = np.mean(rag_scores)

improvement = (
    (avg_rag - avg_llm)
    / avg_llm
) * 100

print("\n")
print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(
    f"Fine Tuned LLM Average      : {avg_llm:.4f}"
)

print(
    f"Fine Tuned + RAG Average    : {avg_rag:.4f}"
)

print(
    f"Improvement                : {improvement:.2f}%"
)

if avg_rag > avg_llm:
    print("\nRAG performed better overall.")
else:
    print("\nFine-tuned model performed better overall.")