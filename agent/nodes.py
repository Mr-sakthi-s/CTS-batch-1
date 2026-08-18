# agent/nodes.py
"""
All agents of the telecom fault-resolution pipeline, consolidated
into one module as plain functions, plus the LangGraph node
wrappers that plug them into the graph.

Layout:

    1. CONFIG                — env-driven paths / models
    2. SHARED RESOURCES      — lazy singletons (embeddings, DBs, LLM)
    3. RCA ENGINE            — from rca_engine.py
    4. DISPATCH AGENT        — from dispatch_agent.py
    5. ESCALATION AGENT      — from escalation_agent.py
    6. MEMORY AGENT          — from memory_agent.py
    7. LANGGRAPH NODES       — thin wrappers over the agents

Heavy dependencies (langchain, chroma, ollama) are imported
lazily inside the resource getters, so the graph can be built,
inspected, and unit-tested without them installed.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .prompts import build_final_rca_prompt, build_pattern_analyst_prompt
from .state import AgentState, RCACandidate


# ==========================================================
# 1. CONFIG
# ==========================================================

# Project root = folder that contains the "agent" package.
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def _first_existing(*paths: str) -> str:
    """
    Return the first path that exists on this machine.
    Falls back to the last path if none exist yet
    (e.g. files created at runtime).
    """
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[-1]


# ----------------------------------------------------------
# ORIGINAL MACHINE PATHS (restored from the original files).
# These are tried FIRST. If they don't exist on the current
# machine, the code automatically falls back to the local
# project's data/ folder — so the same zip runs anywhere.
# Env vars override everything.
# ----------------------------------------------------------

_ORIG_TECHNICIANS_CSV = (
    r"E:\github\CTS-batch-1\RAG\data\technicians.csv"
)

_ORIG_SPARE_PARTS_CSV = (
    r"E:\github\CTS-batch-1\RAG\data\spare_parts.csv"
)

_ORIG_EMBEDDING_MODEL = (
    r"C:\Users\sadik\.cache\huggingface\hub"
    r"\models--sentence-transformers--all-MiniLM-L6-v2"
    r"\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

DATA_DIR = os.getenv(
    "AGENT_DATA_DIR",
    os.path.join(BASE_DIR, "data"),
)

TECHNICIANS_CSV = os.getenv(
    "TECHNICIANS_CSV",
    _first_existing(
        _ORIG_TECHNICIANS_CSV,
        os.path.join(DATA_DIR, "technicians.csv"),
    ),
)

SPARE_PARTS_CSV = os.getenv(
    "SPARE_PARTS_CSV",
    _first_existing(
        _ORIG_SPARE_PARTS_CSV,
        os.path.join(DATA_DIR, "spare_parts.csv"),
    ),
)

RESOLUTION_FILE = os.getenv(
    "RESOLUTION_FILE",
    os.path.join(BASE_DIR, "resolution_history.csv"),
)

VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    os.path.join(BASE_DIR, "vector_db"),
)

# Embedding model: original local snapshot first; if that
# path doesn't exist on this machine, use the HF hub id
# (auto-downloads / resolves from the local HF cache).
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    (
        _ORIG_EMBEDDING_MODEL
        if os.path.exists(_ORIG_EMBEDDING_MODEL)
        else "sentence-transformers/all-MiniLM-L6-v2"
    ),
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "telecom-copilot")

NUM_REGIONS = 10
MAX_RCA_ATTEMPTS = 3

PG_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5432"),
    "dbname": os.getenv("PGDATABASE", "telecom_fault_prediction"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}


# ==========================================================
# 2. SHARED RESOURCES (lazy singletons)
# ==========================================================

@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_knowledge_db():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="telecom_knowledge",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_pattern_db():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="telecom_patterns",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_knowledge_retriever():
    return get_knowledge_db().as_retriever(search_kwargs={"k": 3})


@lru_cache(maxsize=1)
def get_pattern_retriever():
    return get_pattern_db().as_retriever(search_kwargs={"k": 5})


@lru_cache(maxsize=1)
def get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


# ==========================================================
# 3. RCA ENGINE  (was rca_engine.py)
# ==========================================================

def extract_text(response) -> str:
    """Normalize an LLM response object to plain text."""
    if response is None:
        return ""

    content = getattr(response, "content", response)
    if content is None:
        return ""

    if isinstance(content, list):
        result = ""
        for block in content:
            if isinstance(block, dict):
                result += block.get("text", "")
            else:
                result += str(block)
        return result.strip()

    return str(content).strip()


def build_query(ml_output: Dict[str, Any]) -> str:
    return f"""
Severity Type:
{ml_output['severity_type']}

Resource Type:
{ml_output['resource_type']}

Event Types:
{", ".join(ml_output['event_types'])}

Log Features:
{", ".join(ml_output['log_features'])}

Predicted Fault Severity:
{ml_output['predicted_fault_severity']}

Volume:
{ml_output['volume']}
""".strip()


# ----------------------------------------------------------
# SEMANTIC MAPPINGS
# ----------------------------------------------------------

SEVERITY_MAP = {
    "severity_type 1": "Informational Alert",
    "severity_type 2": "Minor Alert",
    "severity_type 3": "Moderate Alert",
    "severity_type 4": "Major Alert",
    "severity_type 5": "Critical Alert",
}

RESOURCE_MAP = {
    "resource_type 1": "Core Router Group",
    "resource_type 2": "Edge Router Group",
    "resource_type 3": "Access Switch Group",
    "resource_type 4": "Aggregation Switch Group",
    "resource_type 5": "Fiber Infrastructure",
    "resource_type 6": "Optical Transmission Equipment",
    "resource_type 7": "Base Station Controller",
    "resource_type 8": "Radio Access Equipment",
    "resource_type 9": "Power Supply Systems",
    "resource_type 10": "Environmental Control Systems",
}

EVENT_MAP = {
    "event_type 1": "Connectivity Warning",
    "event_type 2": "Connectivity Failure",
    "event_type 3": "Link Instability Alert",
    "event_type 4": "Interface Failure Alert",
    "event_type 5": "Network Reachability Alert",
    "event_type 6": "Connection Reset Event",
    "event_type 7": "Packet Loss Alert",
    "event_type 8": "Session Failure Alert",
    "event_type 9": "Service Timeout Alert",
    "event_type 10": "Network Recovery Event",
    "event_type 11": "Traffic Congestion Alert",
    "event_type 12": "Queue Overflow Alert",
    "event_type 13": "Bandwidth Saturation Alert",
    "event_type 14": "High Latency Alert",
    "event_type 15": "Throughput Degradation Alert",
    "event_type 16": "Traffic Spike Alert",
    "event_type 17": "Backhaul Congestion Alert",
    "event_type 18": "Routing Delay Alert",
    "event_type 19": "Core Network Load Alert",
    "event_type 20": "Performance Degradation Alert",
    "event_type 21": "Routing Table Change",
    "event_type 22": "Route Instability Alert",
    "event_type 23": "Route Convergence Delay",
    "event_type 24": "BGP Peer Failure",
    "event_type 25": "OSPF Neighbor Failure",
    "event_type 26": "Route Advertisement Error",
    "event_type 27": "Route Policy Violation",
    "event_type 28": "Network Loop Detection",
    "event_type 29": "Gateway Failure",
    "event_type 30": "Traffic Rerouting Event",
    "event_type 31": "Hardware Health Warning",
    "event_type 32": "Device Failure Alert",
    "event_type 33": "Temperature Warning",
    "event_type 34": "Cooling Failure Alert",
    "event_type 35": "Power Instability Alert",
    "event_type 36": "Battery Failure Alert",
    "event_type 37": "Hardware Restart Event",
    "event_type 38": "Optical Signal Loss",
    "event_type 39": "Fiber Quality Warning",
    "event_type 40": "Physical Infrastructure Failure",
    "event_type 41": "Authentication Failure",
    "event_type 42": "Authorization Failure",
    "event_type 43": "DNS Resolution Failure",
    "event_type 44": "Application Service Failure",
    "event_type 45": "Database Service Failure",
    "event_type 46": "Configuration Error Alert",
    "event_type 47": "Firmware Error Alert",
    "event_type 48": "Security Threat Alert",
    "event_type 49": "Intrusion Detection Alert",
    "event_type 50": "DDoS Suspicion Alert",
    "event_type 51": "Anomaly Detection Alert",
    "event_type 52": "Critical Service Failure",
    "event_type 53": "System-Wide Outage Alert",
}

FEATURE_GROUPS = [
    (1, 50, "Packet Quality Indicators"),
    (51, 100, "Traffic Indicators"),
    (101, 150, "Hardware Indicators"),
    (151, 200, "Power Indicators"),
    (201, 250, "Optical Indicators"),
    (251, 300, "Configuration Indicators"),
    (301, 386, "Security Indicators"),
]


def map_feature(feature: str) -> str:
    match = re.search(r"(\d+)", str(feature))
    if not match:
        return "Unknown Feature Group"

    value = int(match.group(1))
    for low, high, label in FEATURE_GROUPS:
        if low <= value <= high:
            return label

    return "Unknown Feature Group"


def build_semantic_incident(ml_output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "severity": SEVERITY_MAP.get(
            ml_output["severity_type"], "Unknown Severity"
        ),
        "resource": RESOURCE_MAP.get(
            ml_output["resource_type"], "Unknown Resource"
        ),
        "events": [
            EVENT_MAP.get(event, "Unknown Event")
            for event in ml_output["event_types"]
        ],
        "feature_groups": [
            map_feature(feature)
            for feature in ml_output["log_features"]
        ],
        "predicted_fault_severity": ml_output["predicted_fault_severity"],
        "volume": ml_output["volume"],
        "raw_severity": ml_output["severity_type"],
        "raw_resource": ml_output["resource_type"],
        "raw_events": ml_output["event_types"],
        "raw_features": ml_output["log_features"],
    }


def build_semantic_query(semantic_incident: Dict[str, Any]) -> str:
    return f"""
Severity:
{semantic_incident['severity']}

Resource:
{semantic_incident['resource']}

Events:
{", ".join(semantic_incident['events'])}

Feature Groups:
{", ".join(semantic_incident['feature_groups'])}

Predicted Fault Severity:
{semantic_incident['predicted_fault_severity']}

Volume:
{semantic_incident['volume']}
""".strip()


# ----------------------------------------------------------
# RCA RESPONSE PARSING
# ----------------------------------------------------------

def _parse_confidence(value) -> float:
    """Normalize 95%, 95, or 0.95 to a 0.0-1.0 float."""
    if value is None:
        return 0.0
    try:
        value = str(value).strip()
        if "%" in value:
            number = float(value.replace("%", "").strip())
            return max(0.0, min(1.0, number / 100.0))
        number = float(value)
        if number > 1.0:
            number /= 100.0
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return 0.0


def parse_plain_text_rca(
    text: str,
    semantic_incident: Dict[str, Any],
    pattern_analysis: str,
    pattern_context: str = "",
) -> Dict[str, Any]:
    """
    Parse the LLM's plain-text RCA response.

    Supports:
      1. Explicit RCA_1 / RCA_2 / RCA_3 output.
      2. Natural-language single-RCA output.
      3. Historical RAG fallback for remaining candidates.
      4. Pattern-analyst fallback.

    Always returns exactly three ranked candidates or raises.
    """
    if not text:
        raise RuntimeError("LLM returned an empty final RCA response.")

    text = str(text).strip()
    text = re.sub(
        r"```(?:text|plain|markdown)?", "", text, flags=re.IGNORECASE
    )
    text = text.replace("```", "").strip()

    result: Dict[str, Any] = {
        "ranked_causes": [],
        "technical_summary": "",
        "risk_level": "UNKNOWN",
        "semantic_incident": semantic_incident,
        "pattern_analysis": pattern_analysis,
    }

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # -- RISK LEVEL ----------------------------------------
    risk_match = re.search(
        r"(?:RISK[_ ]?LEVEL|RISK)\s*[:\-]\s*([^\n]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if risk_match:
        result["risk_level"] = risk_match.group(1).strip()

    # -- TECHNICAL SUMMARY ---------------------------------
    summary_match = re.search(
        r"""
        TECHNICAL[_ ]?SUMMARY \s*[:\-]\s* (.*?)
        (?=
            \n\s*RCA[_ ]?\d+
            | \n\s*ROOT[_ ]?CAUSE
            | \n\s*RESOLVED[_ ]ROOT[_ ]CAUSE
            | \Z
        )
        """,
        normalized,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )
    if summary_match:
        result["technical_summary"] = summary_match.group(1).strip()

    # -- METHOD 1: EXPLICIT RCA_1 / RCA_2 / RCA_3 ----------
    header_pattern = re.compile(
        r"^\s*RCA[_ \-]?(\d+)\s*[:.\-]?\s*",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    headers = list(header_pattern.finditer(normalized))

    for index, header in enumerate(headers):
        block_start = header.end()
        block_end = (
            headers[index + 1].start()
            if index + 1 < len(headers)
            else len(normalized)
        )
        block = normalized[block_start:block_end].strip()

        root_match = re.search(
            r"""
            ROOT[_ ]?CAUSE \s*[:\-]\s* (.*?)
            (?=
                \n\s*RESOLUTION | \n\s*CONFIDENCE
                | \n\s*EVIDENCE | \Z
            )
            """,
            block,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        resolution_match = re.search(
            r"""
            RESOLUTION \s*[:\-]\s* (.*?)
            (?= \n\s*CONFIDENCE | \n\s*EVIDENCE | \Z )
            """,
            block,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        confidence_match = re.search(
            r"CONFIDENCE\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?%?)",
            block,
            flags=re.IGNORECASE,
        )
        evidence_match = re.search(
            r"EVIDENCE\s*[:\-]\s*(.*)$",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )

        root_cause = root_match.group(1).strip() if root_match else ""
        if root_cause:
            result["ranked_causes"].append({
                "rank": int(header.group(1)),
                "root_cause": root_cause.rstrip("."),
                "resolution": (
                    resolution_match.group(1).strip().rstrip(".")
                    if resolution_match else ""
                ),
                "confidence": _parse_confidence(
                    confidence_match.group(1) if confidence_match else "0"
                ),
                "evidence": (
                    evidence_match.group(1).strip()
                    if evidence_match else ""
                ),
            })

    # -- METHOD 2: NATURAL-LANGUAGE SINGLE RCA -------------
    if len(result["ranked_causes"]) == 0:
        root_match = re.search(
            r"""
            (?: ROOT[_ ]?CAUSE | RESOLVED[_ ]ROOT[_ ]CAUSE )
            \s*[:\-]\s* (.*?)
            (?=
                \n\s* (?: DESCRIPTION | RESOLUTION
                        | CONFIDENCE | EVIDENCE ) \s*[:\-]
                | \Z
            )
            """,
            normalized,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        resolution_match = re.search(
            r"""
            RESOLUTION \s*[:\-]\s* (.*?)
            (?= \n\s* (?: CONFIDENCE | EVIDENCE ) \s*[:\-] | \Z )
            """,
            normalized,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        confidence_match = re.search(
            r"CONFIDENCE\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?%?)",
            normalized,
            flags=re.IGNORECASE,
        )
        evidence_match = re.search(
            r"EVIDENCE\s*[:\-]\s*(.*)$",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if root_match:
            result["ranked_causes"].append({
                "rank": 1,
                "root_cause": root_match.group(1).strip().rstrip("."),
                "resolution": (
                    resolution_match.group(1).strip().rstrip(".")
                    if resolution_match else ""
                ),
                "confidence": _parse_confidence(
                    confidence_match.group(1) if confidence_match else "0"
                ),
                "evidence": (
                    evidence_match.group(1).strip()
                    if evidence_match else ""
                ),
            })

    # -- METHOD 3: HISTORICAL RAG FALLBACK -----------------
    if len(result["ranked_causes"]) < 3:
        print()
        print("LLM returned fewer than 3 RCA candidates.")
        print("Recovering additional candidates from historical RAG evidence...")

        historical_pattern = re.compile(
            r"""
            Resolved\s+Root\s+Cause \s*:\s* (.*?)
            \n\s* Successful\s+Resolution \s*:\s* (.*?)
            (?=
                \n\s* RAW\s+INCIDENT\s+SIGNATURE
                | \n\s* Incident\s+ID
                | \Z
            )
            """,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        historical_matches = historical_pattern.findall(pattern_context)
        print("Historical RCA candidates found:", len(historical_matches))

        existing = {
            c["root_cause"].strip().lower()
            for c in result["ranked_causes"]
        }

        for hist_root, hist_resolution in historical_matches:
            hist_root = hist_root.strip().rstrip(".")
            hist_resolution = hist_resolution.strip().rstrip(".")

            if not hist_root or hist_root.lower() in existing:
                continue

            result["ranked_causes"].append({
                "rank": len(result["ranked_causes"]) + 1,
                "root_cause": hist_root,
                "resolution": hist_resolution,
                "confidence": 0.0,
                "evidence": (
                    "Historical RAG incident with a "
                    "previously successful resolution."
                ),
            })
            existing.add(hist_root.lower())

            if len(result["ranked_causes"]) == 3:
                break

    # -- METHOD 4: PATTERN ANALYST FALLBACK ----------------
    if len(result["ranked_causes"]) < 3:
        pattern_roots = re.findall(
            r"""
            (?: Historical\s+root\s+cause
              | Resolved\s+root\s+cause
              | Root\s+cause )
            \s*[:\-]\s* ([^\n.]+)
            """,
            pattern_analysis,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        existing = {
            c["root_cause"].strip().lower()
            for c in result["ranked_causes"]
        }

        for root in pattern_roots:
            root = root.strip().rstrip(".")
            if not root or root.lower() in existing:
                continue

            result["ranked_causes"].append({
                "rank": len(result["ranked_causes"]) + 1,
                "root_cause": root,
                "resolution": "Review and remediate the identified condition.",
                "confidence": 0.0,
                "evidence": "Identified from historical pattern analysis.",
            })
            existing.add(root.lower())

            if len(result["ranked_causes"]) == 3:
                break

    # -- FINAL VALIDATION ----------------------------------
    if len(result["ranked_causes"]) != 3:
        print()
        print("=" * 100)
        print("RCA PARSER FAILED")
        print("=" * 100)
        print("Expected 3 RCA candidates.")
        print("Detected:", len(result["ranked_causes"]))
        print()
        print("RAW LLM RESPONSE:")
        print(normalized)
        print()
        print("HISTORICAL PATTERN CONTEXT:")
        print(pattern_context)
        print("=" * 100)

        raise RuntimeError(
            "Unable to construct exactly 3 RCA candidates."
        )

    # -- NORMALIZE -----------------------------------------
    for index, candidate in enumerate(result["ranked_causes"], start=1):
        candidate["rank"] = index
        candidate["root_cause"] = str(candidate.get("root_cause", "")).strip()
        candidate["resolution"] = str(candidate.get("resolution", "")).strip()
        candidate["evidence"] = str(candidate.get("evidence", "")).strip()
        candidate["confidence"] = _parse_confidence(
            candidate.get("confidence", 0.0)
        )

    if not result["risk_level"]:
        result["risk_level"] = "UNKNOWN"

    if not result["technical_summary"]:
        result["technical_summary"] = (
            "The incident was analyzed using current telecom telemetry, "
            "domain knowledge, and historical incident patterns."
        )

    return result


def generate_rca_agentic(ml_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full agentic RCA:
        query -> semantic incident -> knowledge RAG ->
        pattern RAG -> pattern analyst -> final RCA -> parse.
    """
    # STEP 1 — raw query
    query = build_query(ml_output)

    # STEP 2 — semantic incident
    semantic_incident = build_semantic_incident(ml_output)

    # STEP 3 — semantic query
    semantic_query = build_semantic_query(semantic_incident)

    # STEP 4 — knowledge retrieval
    try:
        knowledge_docs = get_knowledge_retriever().invoke(semantic_query)
    except Exception as exc:
        print("\nKnowledge Retrieval Error:", exc)
        knowledge_docs = []

    knowledge_context = "\n\n".join(
        doc.page_content for doc in knowledge_docs
    )

    # STEP 5 — historical pattern retrieval
    pattern_query = f"""
RAW INCIDENT:
{query}

SEMANTIC INCIDENT:
{semantic_query}
""".strip()

    try:
        pattern_docs = get_pattern_retriever().invoke(pattern_query)
    except Exception as exc:
        print("\nPattern Retrieval Error:", exc)
        pattern_docs = []

    pattern_context = "\n\n".join(
        doc.page_content for doc in pattern_docs
    )

    print()
    print("=" * 100)
    print("MAPPED INCIDENT")
    print("=" * 100)
    print(json.dumps(semantic_incident, indent=2))

    # STEP 6 — pattern analyst agent
    pattern_prompt = build_pattern_analyst_prompt(
        semantic_query, pattern_context
    )
    try:
        pattern_analysis = extract_text(
            get_llm().invoke(pattern_prompt)
        )
    except Exception as exc:
        print("\nPattern Agent Error:", exc)
        pattern_analysis = ""

    # STEP 7 — final RCA agent
    final_prompt = build_final_rca_prompt(
        query=query,
        semantic_query=semantic_query,
        knowledge_context=knowledge_context,
        pattern_analysis=pattern_analysis,
        pattern_context=pattern_context,
    )

    try:
        final_content = extract_text(get_llm().invoke(final_prompt))
    except Exception as exc:
        print("\nFinal RCA LLM Error:", exc)
        raise RuntimeError("Final RCA generation failed.") from exc

    if not final_content:
        raise RuntimeError("LLM returned an empty final RCA response.")

    result = parse_plain_text_rca(
        final_content,
        semantic_incident,
        pattern_analysis,
        pattern_context,
    )

    return {
        "ranked_causes": result["ranked_causes"],
        "technical_summary": result["technical_summary"],
        "risk_level": result["risk_level"],
        "semantic_incident": semantic_incident,
        "pattern_analysis": pattern_analysis,
        "knowledge_context": knowledge_context,
        "pattern_context": pattern_context,
    }


# ==========================================================
# 4. DISPATCH AGENT  (was dispatch_agent.py)
# ==========================================================

def derive_region(location: str) -> str:
    """location 118 -> region_8, location 662 -> region_2."""
    match = re.search(r"\d+", str(location))
    if not match:
        raise ValueError(
            f"Could not extract location number from: {location}"
        )
    return f"region_{int(match.group()) % NUM_REGIONS}"


def load_reference_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    technicians = pd.read_csv(TECHNICIANS_CSV)
    spare_parts = pd.read_csv(SPARE_PARTS_CSV)

    # Normalize booleans in case the CSV has strings.
    if technicians["available"].dtype == object:
        technicians["available"] = (
            technicians["available"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    return technicians, spare_parts


def _region_num(region: str) -> int:
    return int(region.split("_")[1])


def _region_distance(region_a: str, region_b: str) -> int:
    """Circular distance between two regions."""
    diff = abs(_region_num(region_a) - _region_num(region_b))
    return min(diff, NUM_REGIONS - diff)


def _regions_by_distance(home_region: str) -> List[str]:
    regions = [f"region_{i}" for i in range(NUM_REGIONS)]
    return sorted(
        regions,
        key=lambda region: _region_distance(region, home_region),
    )


def find_technician_in_region(region, resource_type, technicians):
    """Priority: region -> skill -> available -> lowest load."""
    candidates = technicians[
        (technicians["region"] == region)
        & (technicians["skill_type"] == resource_type)
        & (technicians["available"] == True)  # noqa: E712
    ].sort_values("current_load")

    if candidates.empty:
        return None
    return candidates.iloc[0]


def part_in_stock(region, resource_type, spare_parts) -> bool:
    mask = (
        (spare_parts["region"] == region)
        & (spare_parts["part_type"] == resource_type)
    )
    rows = spare_parts.loc[mask]
    if rows.empty:
        return False
    return float(rows.iloc[0]["stock_count"]) > 0


def reserve_part(region, resource_type, spare_parts) -> None:
    mask = (
        (spare_parts["region"] == region)
        & (spare_parts["part_type"] == resource_type)
    )
    spare_parts.loc[mask, "stock_count"] -= 1


def find_best_dispatch(home_region, resource_type, technicians, spare_parts):
    """
    Pass 1: technician + spare in the same region.
    Pass 2: technician even without a spare.
    """
    search_order = _regions_by_distance(home_region)

    for region in search_order:
        technician = find_technician_in_region(
            region, resource_type, technicians
        )
        if technician is not None and part_in_stock(
            region, resource_type, spare_parts
        ):
            return (
                technician,
                region,
                True,
                _region_distance(region, home_region),
            )

    for region in search_order:
        technician = find_technician_in_region(
            region, resource_type, technicians
        )
        if technician is not None:
            return (
                technician,
                region,
                False,
                _region_distance(region, home_region),
            )

    return None, None, False, None


def assign_dispatch(fault, technicians, spare_parts) -> Dict[str, Any]:
    home_region = derive_region(fault["location"])
    resource_type = fault["resource_type"]

    technician, source_region, part_available, distance = (
        find_best_dispatch(
            home_region, resource_type, technicians, spare_parts
        )
    )

    escalation = None

    if technician is None:
        escalation = escalate(
            {"ticket_id": str(fault["id"])},
            reason=(
                "No available technician with matching skill "
                f"({resource_type}) in any region."
            ),
        )
        status = "ESCALATED"

    elif not part_available:
        status = "part_shortage_flagged"

    else:
        status = (
            "assigned" if distance == 0 else "assigned_cross_region"
        )
        technicians.loc[
            technicians["technician_id"] == technician["technician_id"],
            "current_load",
        ] += 1
        reserve_part(source_region, resource_type, spare_parts)

    result: Dict[str, Any] = {
        "ticket_id": str(fault["id"]),
        "location": fault["location"],
        "region": home_region,
        "resource_type": resource_type,
        "fault_severity": int(fault["fault_severity"]),
        "root_cause": fault.get("root_cause"),
        "recommended_solution": fault.get("recommended_solution"),
        "status": status,
        "technician": None,
        "spare_part": None,
        "escalation": escalation,
    }

    if technician is not None:
        result["technician"] = {
            "technician_id": technician["technician_id"],
            "technician_name": technician["technician_name"],
            "region": technician["region"],
            "cross_region": distance is not None and distance > 0,
            "distance_from_fault": (
                int(distance) if distance is not None else None
            ),
            "skill_type": technician["skill_type"],
            "current_load_after_assignment": int(
                technician["current_load"]
                + (1 if status.startswith("assigned") else 0)
            ),
        }

    result["spare_part"] = {
        "part_type": resource_type,
        "sourced_region": source_region,
        "available": part_available,
    }

    return result


# ==========================================================
# 5. ESCALATION AGENT  (was escalation_agent.py)
# ==========================================================

def escalate(
    ticket: Dict[str, Any],
    reason: str = "Issue not resolved after all RCA recommendations.",
) -> Dict[str, Any]:
    """Escalate unresolved incidents to the human NOC team."""
    return {
        "ticket_id": str(ticket.get("ticket_id", "UNKNOWN")),
        "status": "ESCALATED",
        "reason": reason,
        "assigned_group": "NOC_ENGINEERING_TEAM",
    }


# ==========================================================
# 6. MEMORY AGENT  (was memory_agent.py)
# ==========================================================

def save_resolution(
    ml_output: Dict[str, Any],
    semantic_incident: Dict[str, Any],
    root_cause: str,
    successful_action: str,
) -> None:
    """Persist a successful resolution to CSV + the pattern vector DB."""
    raw_signature = (
        f"Severity Type: {ml_output['severity_type']}\n"
        f"Resource Type: {ml_output['resource_type']}\n"
        f"Event Types: {','.join(ml_output['event_types'])}\n"
        f"Log Features: {','.join(ml_output['log_features'])}"
    )

    semantic_signature = (
        f"Severity: {semantic_incident['severity']}\n"
        f"Resource: {semantic_incident['resource']}\n"
        f"Events: {','.join(semantic_incident['events'])}\n"
        f"Feature Groups: "
        f"{','.join(semantic_incident['feature_groups'])}"
    )

    document = f"""
RAW INCIDENT SIGNATURE:
{raw_signature}

SEMANTIC INCIDENT SIGNATURE:
{semantic_signature}

Resolved Root Cause:
{root_cause}

Successful Resolution:
{successful_action}
""".strip()

    # -- CSV -----------------------------------------------
    row = pd.DataFrame([{
        "raw_incident_signature": raw_signature.replace("\n", " | "),
        "semantic_incident_signature": semantic_signature.replace(
            "\n", " | "
        ),
        "root_cause": root_cause,
        "successful_resolution": successful_action,
    }])

    if os.path.exists(RESOLUTION_FILE):
        existing = pd.read_csv(RESOLUTION_FILE)
        pd.concat([existing, row], ignore_index=True).to_csv(
            RESOLUTION_FILE, index=False
        )
    else:
        row.to_csv(RESOLUTION_FILE, index=False)

    # -- VECTOR DB -----------------------------------------
    get_pattern_db().add_texts(
        texts=[document],
        metadatas=[{
            "severity_type": ml_output["severity_type"],
            "resource_type": ml_output["resource_type"],
            "root_cause": root_cause,
            "source": "successful_resolution",
        }],
    )

    print()
    print("=" * 90)
    print("SELF-LEARNING MEMORY UPDATED")
    print("=" * 90)
    print(document)


# ==========================================================
# 7. LANGGRAPH NODES
# ==========================================================
# Each node takes the full AgentState and returns a partial
# state update. LangGraph merges the update into the state.
# ==========================================================

def rca_node(state: AgentState) -> Dict[str, Any]:
    """Run the agentic RCA engine and open a ticket."""
    ml_output = state["ml_output"]
    fault = state.get("fault", {})

    rca = generate_rca_agentic(ml_output)
    ranked: List[RCACandidate] = rca["ranked_causes"]

    ticket = {
        "ticket_id": str(fault.get("id", "UNKNOWN")),
        "status": "IN_PROGRESS",
        "attempt": 0,
        "ranked_causes": ranked,
    }

    return {
        "semantic_incident": rca["semantic_incident"],
        "knowledge_context": rca["knowledge_context"],
        "pattern_context": rca["pattern_context"],
        "pattern_analysis": rca["pattern_analysis"],
        "ranked_causes": ranked,
        "technical_summary": rca["technical_summary"],
        "risk_level": rca["risk_level"],
        "ticket": ticket,
        "attempt": 0,
        "current_candidate": ranked[0],
        "status": "IN_PROGRESS",
    }


def dispatch_node(state: AgentState) -> Dict[str, Any]:
    """Assign a technician + spare part for the current RCA hypothesis."""
    candidate = state.get("current_candidate") or {}

    fault = dict(state["fault"])
    fault["root_cause"] = candidate.get("root_cause")
    fault["recommended_solution"] = candidate.get("resolution")

    technicians, spare_parts = load_reference_data()
    dispatch_result = assign_dispatch(fault, technicians, spare_parts)

    update: Dict[str, Any] = {"dispatch_result": dispatch_result}

    # No technician anywhere -> the dispatch agent already escalated.
    if dispatch_result["status"] == "ESCALATED":
        ticket = dict(state["ticket"])
        ticket["status"] = "ESCALATED"
        update["ticket"] = ticket
        update["status"] = "ESCALATE"
        update["escalation"] = dispatch_result["escalation"]

    return update


def verify_node(state: AgentState) -> Dict[str, Any]:
    """
    Determine whether the attempted resolution actually fixed
    the incident.

    Simulation mode: consumes one boolean from
    ``state["feedback_queue"]`` per attempt.

    Production mode: compile the graph with
    ``interrupt_before=["verify"]`` and a checkpointer, pause
    here, and resume with ``{"fixed": True/False}`` supplied by
    the technician or by network monitoring.
    """
    queue = list(state.get("feedback_queue", []))

    if queue:
        fixed = bool(queue.pop(0))
    else:
        fixed = bool(state.get("fixed", False))

    return {"fixed": fixed, "feedback_queue": queue}


def feedback_node(state: AgentState) -> Dict[str, Any]:
    """
    RCA hypothesis feedback loop (was feedback_agent.py).

    Success               -> CLOSED
    Failure, more hypos   -> RETRY with next ranked candidate
    All hypotheses failed -> ESCALATE
    """
    ticket = dict(state["ticket"])

    # -- SUCCESS -------------------------------------------
    if state.get("fixed"):
        ticket["status"] = "CLOSED"
        return {"ticket": ticket, "status": "CLOSED"}

    # -- FAILED ATTEMPT ------------------------------------
    attempt = state.get("attempt", 0) + 1
    ticket["attempt"] = attempt
    ranked = ticket["ranked_causes"]

    # -- MORE RCA HYPOTHESES AVAILABLE ---------------------
    if attempt < min(len(ranked), MAX_RCA_ATTEMPTS):
        return {
            "ticket": ticket,
            "attempt": attempt,
            "current_candidate": ranked[attempt],
            "status": "RETRY",
        }

    # -- ALL HYPOTHESES FAILED -----------------------------
    ticket["status"] = "ESCALATED"
    return {"ticket": ticket, "attempt": attempt, "status": "ESCALATE"}


def memory_node(state: AgentState) -> Dict[str, Any]:
    """Persist the successful root cause + resolution (self-learning)."""
    candidate = state.get("current_candidate") or {}

    save_resolution(
        ml_output=state["ml_output"],
        semantic_incident=state["semantic_incident"],
        root_cause=candidate.get("root_cause", ""),
        successful_action=candidate.get("resolution", ""),
    )

    return {"memory_saved": True, "status": "CLOSED"}


def escalation_node(state: AgentState) -> Dict[str, Any]:
    """Hand the ticket to the human NOC engineering team."""
    # If dispatch already escalated (no technician), keep that reason.
    if state.get("escalation"):
        return {"status": "ESCALATED"}

    escalation = escalate(
        state["ticket"],
        reason="Issue not resolved after all RCA recommendations.",
    )

    return {"escalation": escalation, "status": "ESCALATED"}
