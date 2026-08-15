import os
import shutil

import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_RAG_DIR, "data", "combined_data.csv")
_PERSIST_DIR = os.path.join(_RAG_DIR, "vector_db")
_COLLECTION_NAME = "telecom_patterns"


def main():

    # ==========================================
    # CLEAN STALE CHROMA STATE BEFORE REBUILD
    # ==========================================
    # The persisted HNSW index can become unreadable after repeated ingest runs.
    # Remove only the Chroma persistence directory so the next run rebuilds a
    # clean collection instead of reusing a stale/broken index.
    if os.path.isdir(_PERSIST_DIR):
        print(f"Removing stale Chroma persistence directory: {_PERSIST_DIR}")
        shutil.rmtree(_PERSIST_DIR)

    # ==========================================
    # LOAD CSV
    # ==========================================

    df = pd.read_csv(_DATA_PATH)

    print(f"Rows Found: {len(df)}")

    # ==========================================
    # CREATE DOCUMENTS
    # ==========================================

    documents = []

    for _, row in df.iterrows():

        content = f"""
Incident ID: {row['id']}

Severity Category:
{row['severity_category']}

Event Categories:
{row['event_categories']}

Resource Categories:
{row['resource_categories']}

Log Feature Groups:
{row['log_feature_groups']}

Event Count:
{row['event_count']}

Total Log Volume:
{row['total_log_volume']}

Mean Log Volume:
{row['mean_log_volume']}

Unique Log Features:
{row['unique_log_features']}

Historical Root Cause:
{row['root_cause_description']}

Historical Solution:
{row['solution_description']}
"""

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "incident_id": int(row["id"])
                }
            )
        )

    print(f"Pattern Documents: {len(documents)}")

    # ==========================================
    # EMBEDDINGS
    # ==========================================
    # multi_process=True spreads encoding across CPU cores instead of one,
    # and a larger batch_size cuts down per-batch overhead. Both matter a
    # lot at 18k+ rows on CPU. multi_process requires the
    # `if __name__ == "__main__":` guard below on Windows.

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        multi_process=True,
        show_progress=True,
        encode_kwargs={"batch_size": 128},
    )

    # ==========================================
    # STORE IN CHROMA
    # ==========================================

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=_COLLECTION_NAME,
        persist_directory=_PERSIST_DIR,
    )

    print(f"Pattern Database Indexed Successfully -> {_COLLECTION_NAME} @ {_PERSIST_DIR}")


if __name__ == "__main__":
    main()