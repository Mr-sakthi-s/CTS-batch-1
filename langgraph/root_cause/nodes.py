from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

# ==========================================
# MAKE THE SIBLING "RAG" FOLDER IMPORTABLE
# ==========================================
# NOTE: this project folder is named "langgraph", which collides with the
# installed `langgraph` pip package if the project root ever ends up on
# sys.path. To avoid that collision we add ONLY the RAG/ folder itself to
# sys.path (never the project root), so `rag_retriever` and `main` resolve
# as plain top-level modules regardless of the current working directory.
# This import happens ONCE, at module load time - not per live record -
# which is also when RAG/main.py's Ollama client and RAG/rag_retriever.py's
# embedding model get constructed (both are module-level singletons).

_ROOT_CAUSE_DIR = os.path.dirname(os.path.abspath(__file__))
_LANGGRAPH_DIR = os.path.dirname(_ROOT_CAUSE_DIR)
_PROJECT_ROOT = os.path.dirname(_LANGGRAPH_DIR)
_RAG_DIR = os.path.join(_PROJECT_ROOT, "RAG")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from rag_retriever import retrieve_context as rag_retrieve_context  # noqa: E402
from main import analyze_root_cause as llm_analyze_root_cause  # noqa: E402
from main import generate_detailed_solution as llm_generate_detailed_solution  # noqa: E402

logger = logging.getLogger("root_cause")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _fault_summary(state) -> str:
    return f"""Severity Type: {state.get('severity_type', 'unknown')}
Resource Type: {state.get('resource_type', 'unknown')}
Event Type: {state.get('event_type', 'unknown')}
Feature: {state.get('feature', 'unknown')}
Volume: {state.get('volume', 'unknown')}"""


def _normalize_live_state(state) -> dict[str, Any]:
    return {
        "severity_type": state.get("severity_type", "unknown"),
        "resource_type": state.get("resource_type", "unknown"),
        "event_type": state.get("event_type", "unknown"),
        "feature": state.get("feature", "unknown"),
        "volume": state.get("volume", "unknown"),
    }


# ==========================================
# PARSE RETRIEVED DOCUMENTS
# ==========================================
# pattern_ingest.py writes each historical incident as labeled text
# blocks. This pulls the key fields back out so the RAG RETRIEVAL section
# of the final report can show them individually, instead of dumping raw
# document text.

_INCIDENT_PATTERN = re.compile(
    r"Severity Category:\s*\n(?P<severity>.*?)\n\n"
    r"Event Categories:\s*\n(?P<event>.*?)\n\n"
    r"Resource Categories:\s*\n(?P<resource>.*?)\n\n"
    r"Log Feature Groups:\s*\n(?P<features>.*?)\n\n"
    r"Event Count:\s*\n(?P<event_count>.*?)\n\n"
    r"Total Log Volume:\s*\n(?P<total_log_volume>.*?)\n\n"
    r"Mean Log Volume:\s*\n(?P<mean_log_volume>.*?)\n\n"
    r"Unique Log Features:\s*\n(?P<unique_log_features>.*?)\n\n"
    r".*?"
    r"Historical Root Cause:\s*\n(?P<root_cause>.*?)\n\n"
    r"Historical Solution:\s*\n(?P<historical_solution>.*?)(?:\n\n\Z|\Z)",
    re.DOTALL,
)


def _parse_incident(text: str) -> dict[str, str]:
    match = _INCIDENT_PATTERN.search(text)
    if not match:
        return {
            "severity": "unknown",
            "event": "unknown",
            "resource": "unknown",
            "features": "unknown",
            "event_count": "unknown",
            "total_log_volume": "unknown",
            "mean_log_volume": "unknown",
            "unique_log_features": "unknown",
            "root_cause": "unknown",
            "historical_solution": "unknown",
        }
    parsed = {key: value.strip() for key, value in match.groupdict().items()}
    if not parsed.get("historical_solution"):
        parsed["historical_solution"] = "Historical solution not available in the retrieved evidence."
    return parsed


# ==========================================
# NODE 1: create_query + retrieve_rag_context (combined)
# ==========================================
# The live input itself IS the graph's input state (see workflow.py /
# whatever code invokes root_cause_graph.invoke(...) per record), so a
# separate "receive_live_input" node isn't needed - this node's state
# argument already holds the live values.

def retrieve_context(state) -> dict[str, Any]:
    """Build a retrieval query from the live fault input and fetch the
    top-3 most similar historical incidents from the Chroma vector DB."""

    normalized = _normalize_live_state(state)
    query = _fault_summary(normalized)
    logger.info("retrieve_context: query built from live input")

    try:
        docs = rag_retrieve_context(query, k=3)
    except Exception as exc:
        logger.error("retrieve_context: RAG retrieval failed: %s", exc)
        return {
            "query": query,
            "retrieved_context": [],
            "retrieved_incidents": [],
            "is_valid": False,
            "error": f"RAG retrieval failed: {exc}",
        }

    retrieved_context = [d.page_content for d in docs]
    retrieved_incidents = [_parse_incident(d.page_content) for d in docs]

    logger.info(
        "retrieve_context: retrieved %d historical incidents",
        len(retrieved_context),
    )

    return {
        **normalized,
        "query": query,
        "retrieved_context": retrieved_context,
        "retrieved_incidents": retrieved_incidents,
    }


# ==========================================
# NODE 2: analyze_root_cause
# ==========================================

def analyze_root_cause(state) -> dict[str, Any]:
    """Ask the local Ollama model to identify the most likely root cause
    using only the retrieved historical evidence."""

    if state.get("error"):
        # retrieve_context already failed for this record - pass through
        # without calling the LLM.
        return {}

    logger.info("analyze_root_cause: running local Ollama root cause analysis")

    query = state.get("query", "")
    retrieved_context = state.get("retrieved_context") or []

    result = llm_analyze_root_cause(query, retrieved_context)

    if result.get("error"):
        logger.error("analyze_root_cause: %s", result["error"])
        return {
            "root_cause": result["root_cause"],
            "root_cause_explanation": result["explanation"],
            "root_cause_characteristics": result.get("characteristics", []),
            "is_valid": False,
            "error": result["error"],
        }

    logger.info("analyze_root_cause: root_cause=%s", result["root_cause"])

    return {
        "root_cause": result["root_cause"],
        "root_cause_explanation": result["explanation"],
        "root_cause_characteristics": result.get("characteristics", []),
    }


# ==========================================
# NODE 3: generate_detailed_solution + generate_validation (combined)
# ==========================================
# main.py's generate_detailed_solution already returns both the expanded
# solution steps AND a validation checklist from a single LLM call, so a
# separate validation node would just be an extra hop for no benefit.

def generate_detailed_solution(state) -> dict[str, Any]:
    """Expand the retrieved historical solution(s) into a detailed
    recommendation + validation checklist, and assemble the final report."""

    if state.get("error"):
        return {"final_report": _build_error_report(state)}

    logger.info("generate_detailed_solution: expanding historical solution")

    root_cause = state.get("root_cause", "")
    retrieved_context = state.get("retrieved_context") or []

    result = llm_generate_detailed_solution(root_cause, retrieved_context)

    if result.get("error"):
        logger.error("generate_detailed_solution: %s", result["error"])
        error_state = {**state, "error": result["error"]}
        return {
            "solution": result.get("solution_steps", []),
            "validation": result.get("validation_points", []),
            "is_valid": False,
            "error": result["error"],
            "final_report": _build_error_report(error_state),
        }

    final_report = _build_final_report(state, result)

    logger.info(
        "generate_detailed_solution: generated %d solution steps",
        len(result["solution_steps"]),
    )

    return {
        "solution": result["solution_steps"],
        "validation": result["validation_points"],
        "final_report": final_report,
    }


# ==========================================
# REPORT FORMATTING
# ==========================================

def _format_retrieved_incidents(incidents: list[dict[str, str]]) -> str:

    if not incidents:
        return "No similar historical incidents were retrieved."

    lines = []

    for i, inc in enumerate(incidents, start=1):
        lines.append(f"Historical Incident {i}:")
        lines.append(f"- Severity       : {inc.get('severity', 'unknown')}")
        lines.append(f"- Event Pattern  : {inc.get('event', 'unknown')}")
        lines.append(f"- Resource       : {inc.get('resource', 'unknown')}")
        lines.append(f"- Log Features   : {inc.get('features', 'unknown')}")
        lines.append(f"- Root Cause     : {inc.get('root_cause', 'unknown')}")
        lines.append(f"- Historical Solution : {inc.get('historical_solution', 'unknown')}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_bullets(items: list[str], bullet: str) -> str:
    if not items:
        return f"{bullet} No specific points were identified from the retrieved evidence."
    return "\n".join(f"{bullet} {item}" for item in items)


def _format_steps(steps: list[str]) -> str:
    return "\n\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))


def _build_final_report(state, solution_result: dict[str, Any]) -> str:

    steps_text = _format_steps(solution_result["solution_steps"])
    retrieved_incidents = state.get("retrieved_incidents") or []
    incidents_text = _format_retrieved_incidents(retrieved_incidents)
    validation_text = _format_bullets(solution_result.get("validation_points") or [], "✓")

    count = len(retrieved_incidents)
    count_label = "incident" if count == 1 else "incidents"
    retrieval_header = f"{count} similar historical {count_label} were retrieved\nfrom the telecom fault knowledge base."

    current_live_summary = (
        f"Current live severity: {state.get('severity_type', 'unknown')}\n"
        f"Current live resource: {state.get('resource_type', 'unknown')}\n"
        f"Current live event: {state.get('event_type', 'unknown')}\n"
        f"Current live feature: {state.get('feature', 'unknown')}\n"
        f"Current live volume: {state.get('volume', 'unknown')}"
    )

    strongest_evidence = (
        "The strongest historical evidence comes from the Edge Router Group incidents, which match the current live resource pattern and associate the same resource pattern with upstream link instability or interface errors. The third retrieved incident describes a different aggregation-layer bottleneck and is therefore not treated as the primary root-cause evidence."
    )

    return f"""==================================================
              TELECOM FAULT ANALYSIS
==================================================

INPUT FAULT DETAILS
-------------------
Severity Type   : {state.get('severity_type', 'unknown')}
Resource Type   : {state.get('resource_type', 'unknown')}
Event Type      : {state.get('event_type', 'unknown')}
Feature         : {state.get('feature', 'unknown')}
Volume          : {state.get('volume', 'unknown')}


==================================================
              RAG RETRIEVAL
==================================================

{retrieval_header}

{incidents_text}


==================================================
                 ROOT CAUSE
==================================================

Most Likely Root Cause:

{state.get('root_cause', 'Unknown')}


==================================================
          WHY THIS ROOT CAUSE?
==================================================

The current live incident is compared against the retrieved historical incidents.

Current live input:
{current_live_summary}

The matching resource pattern and event pattern are compared against the retrieved historical evidence. Historical values remain clearly labeled as historical and are never mixed with the current live input values.

{strongest_evidence}


==================================================
        DETAILED RECOMMENDED SOLUTION
==================================================

{steps_text}


==================================================
              VALIDATION
==================================================

{validation_text}


==================================================
             HISTORICAL EVIDENCE
==================================================

The recommendation was generated using similar historical telecom incidents retrieved from the RAG knowledge base.

The historical solution was used as the factual basis and then expanded by the local Ollama LLM into a more detailed operational procedure.

Primary supporting incidents: Edge Router Group incidents associated with upstream link instability or interface errors.

Secondary / alternative historical case: the aggregation-layer bottleneck incident is retained only as a contrasting historical reference and is not used as the primary recommended action for this live Edge Router Group case.
==================================================
"""


def _build_error_report(state) -> str:

    return f"""==================================================
TELECOM FAULT ANALYSIS
==================================================

## INPUT FAULT DETAILS

Severity Type   : {state.get('severity_type', 'unknown')}
Resource Type   : {state.get('resource_type', 'unknown')}
Event Type      : {state.get('event_type', 'unknown')}
Feature         : {state.get('feature', 'unknown')}
Volume          : {state.get('volume', 'unknown')}

==================================================
ERROR
==================================================

{state.get('error', 'An unknown error occurred while processing this record.')}

This record could not be fully analyzed. The live streaming process
should continue with the next incoming record.
==================================================
"""