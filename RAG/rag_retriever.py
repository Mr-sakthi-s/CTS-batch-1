from __future__ import annotations

import hashlib
import json
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# ENV
# ==========================================

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_RAG_DIR, ".env"))

# ==========================================
# CONFIG
# ==========================================

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_PERSIST_DIR = os.path.join(_RAG_DIR, "vector_db")
_COLLECTION_NAME = "telecom_patterns"

_embeddings = None
_db = None


def _get_db():
    """Create and cache the Chroma connection once for the process."""
    global _embeddings, _db

    if _db is not None:
        return _db

    if not os.path.isdir(_PERSIST_DIR):
        raise FileNotFoundError(
            f"RAG vector database folder not found at {_PERSIST_DIR}. "
            "Run the CSV ingestion step once before running retrieval."
        )

    _embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    _db = Chroma(
        collection_name=_COLLECTION_NAME,
        persist_directory=_PERSIST_DIR,
        embedding_function=_embeddings,
    )
    return _db


def build_live_query_from_record(record: dict) -> str:
    """Create a retrieval query using field names, not raw numeric tokens."""
    if not isinstance(record, dict):
        return ""

    severity_type = record.get("severity_type", "unknown")
    resource_type = record.get("resource_type", "unknown")
    event_type = record.get("event_type", "unknown")
    feature = record.get("feature", "unknown")
    volume = record.get("volume", "unknown")

    return (
        "Severity: " + str(severity_type) + "\n"
        "Resource Type: " + str(resource_type) + "\n"
        "Event Type: " + str(event_type) + "\n"
        "Feature: " + str(feature) + "\n"
        "Volume: " + str(volume)
    )


def _live_record_signature(record: dict) -> str:
    payload = {
        "severity_type": record.get("severity_type"),
        "resource_type": record.get("resource_type"),
        "event_type": record.get("event_type"),
        "feature": record.get("feature"),
        "volume": record.get("volume"),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_live_record(record: dict) -> bool:
    """Store a unique live-generated incident in the existing Chroma collection.

    This adds only the new record and reuses the existing collection, embedding
    model, and persisted Chroma database without rebuilding the full dataset.
    """
    if not isinstance(record, dict):
        return False

    normalized = {
        "severity_type": record.get("severity_type"),
        "resource_type": record.get("resource_type"),
        "event_type": record.get("event_type"),
        "feature": record.get("feature"),
        "volume": record.get("volume"),
    }

    if not any(value not in (None, "", "unknown") for value in normalized.values()):
        return False

    db = _get_db()
    signature = _live_record_signature(normalized)

    try:
        existing = db._collection.get(include=["metadatas"])
    except Exception:
        existing = {"metadatas": []}

    for metadata in existing.get("metadatas", []) or []:
        if isinstance(metadata, dict) and metadata.get("signature") == signature:
            return False

    content = (
        "Severity Type:\n"
        f"{normalized.get('severity_type', 'unknown')}\n\n"
        "Event Type:\n"
        f"{normalized.get('event_type', 'unknown')}\n\n"
        "Resource Type:\n"
        f"{normalized.get('resource_type', 'unknown')}\n\n"
        "Feature:\n"
        f"{normalized.get('feature', 'unknown')}\n\n"
        "Volume:\n"
        f"{normalized.get('volume', 'unknown')}\n\n"
        "Source Type:\nlive_generated\n\n"
        "Record Status:\ngenerated\n\n"
        "Historical Root Cause:\nProvisional live-generated incident. No sufficiently similar historical incident was found.\n\n"
        "Historical Solution:\nProvisional recommendation generated for this live record and stored for future retrieval."
    )

    live_id = f"live_{signature[:12]}"
    doc = Document(
        page_content=content,
        metadata={
            "incident_id": live_id,
            "source_type": "live_generated",
            "record_status": "generated",
            "signature": signature,
            "source": "live_stream",
        },
    )

    try:
        db.add_documents([doc])
        return True
    except Exception:
        return False


def retrieve_context(query: str, k: int = 3):
    """Return up to k unique, relevant historical incident documents."""
    if not query or not query.strip():
        return []

    db = _get_db()
    candidate_count = max(k * 5, 10)
    docs = db.similarity_search(query, k=candidate_count)

    unique_docs = []
    seen = set()

    for doc in docs:
        candidate_text = (doc.page_content or "").strip()
        normalized_text = re.sub(r"\s+", " ", candidate_text)
        incident_key = (
            doc.metadata.get("incident_id")
            if isinstance(doc.metadata, dict) and doc.metadata.get("incident_id") is not None
            else normalized_text
        )

        if incident_key in seen:
            continue

        seen.add(incident_key)
        unique_docs.append(doc)

        if len(unique_docs) >= k:
            break

    return unique_docs