import json
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()


# ==========================================================
# PATHS
# ==========================================================


VECTOR_DB_PATH=r"E:\github\CTS-batch-1\vector_db"


# ==========================================================
# EMBEDDINGS
# ==========================================================

EMBEDDING_MODEL = (
    r"C:\Users\sadik\.cache\huggingface\hub"
    r"\models--sentence-transformers--all-MiniLM-L6-v2"
    r"\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ==========================================================
# KNOWLEDGE VECTOR DB
# ==========================================================

knowledge_db = Chroma(
    collection_name="telecom_knowledge",
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

knowledge_retriever = knowledge_db.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ==========================================================
# HISTORICAL PATTERN VECTOR DB
# ==========================================================

pattern_db = Chroma(
    collection_name="telecom_patterns",
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

pattern_retriever = pattern_db.as_retriever(
    search_kwargs={
        "k": 5
    }
)


# ==========================================================
# OLLAMA
# ==========================================================

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "telecom-copilot:latest"

# Explicit client so FastAPI/Uvicorn always targets the
# same local Ollama HTTP server.
ollama_client = ollama.Client(
    host=OLLAMA_HOST
)


def check_ollama_connection():
    """
    Verify that the local Ollama HTTP API is reachable and that
    the required model is available.
    """
    models = ollama_client.list()

    available_models = [
        model.model
        for model in getattr(models, "models", [])
    ]

    if OLLAMA_MODEL not in available_models:
        raise RuntimeError(
            f"Ollama is reachable, but model '{OLLAMA_MODEL}' "
            f"was not found. Available models: {available_models}"
        )

    return True


def call_ollama(prompt):
    """
    Call telecom-copilot directly through the Ollama HTTP API.
    """
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 512
        }
    )

    content = (
        response
        .get("message", {})
        .get("content", "")
    )

    if content is None:
        return ""

    return str(content).strip()


# ==========================================================
# EXTRACT RESPONSE TEXT
# ==========================================================

def extract_text(response):

    if response is None:
        return ""

    content = getattr(
        response,
        "content",
        response
    )

    if content is None:
        return ""

    # Handle list-style content
    if isinstance(content, list):

        result = ""

        for block in content:

            if isinstance(block, dict):

                result += block.get(
                    "text",
                    ""
                )

            else:

                result += str(
                    block
                )

        return result.strip()

    return str(
        content
    ).strip()


# ==========================================================
# BUILD INCIDENT QUERY
# ==========================================================

def build_query(
    ml_output
):

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


# ==========================================================
# BUILD SEMANTIC INCIDENT
# ==========================================================

def build_semantic_incident(
    ml_output
):

    # ======================================================
    # SEVERITY MAP
    # ======================================================

    severity_map = {

        "severity_type 1":
            "Informational Alert",

        "severity_type 2":
            "Minor Alert",

        "severity_type 3":
            "Moderate Alert",

        "severity_type 4":
            "Major Alert",

        "severity_type 5":
            "Critical Alert"
    }


    # ======================================================
    # RESOURCE MAP
    # ======================================================

    resource_map = {

        "resource_type 1":
            "Core Router Group",

        "resource_type 2":
            "Edge Router Group",

        "resource_type 3":
            "Access Switch Group",

        "resource_type 4":
            "Aggregation Switch Group",

        "resource_type 5":
            "Fiber Infrastructure",

        "resource_type 6":
            "Optical Transmission Equipment",

        "resource_type 7":
            "Base Station Controller",

        "resource_type 8":
            "Radio Access Equipment",

        "resource_type 9":
            "Power Supply Systems",

        "resource_type 10":
            "Environmental Control Systems"
    }


    # ======================================================
    # EVENT MAP
    # ======================================================

    event_map = {

        "event_type 1":
            "Connectivity Warning",

        "event_type 2":
            "Connectivity Failure",

        "event_type 3":
            "Link Instability Alert",

        "event_type 4":
            "Interface Failure Alert",

        "event_type 5":
            "Network Reachability Alert",

        "event_type 6":
            "Connection Reset Event",

        "event_type 7":
            "Packet Loss Alert",

        "event_type 8":
            "Session Failure Alert",

        "event_type 9":
            "Service Timeout Alert",

        "event_type 10":
            "Network Recovery Event",

        "event_type 11":
            "Traffic Congestion Alert",

        "event_type 12":
            "Queue Overflow Alert",

        "event_type 13":
            "Bandwidth Saturation Alert",

        "event_type 14":
            "High Latency Alert",

        "event_type 15":
            "Throughput Degradation Alert",

        "event_type 16":
            "Traffic Spike Alert",

        "event_type 17":
            "Backhaul Congestion Alert",

        "event_type 18":
            "Routing Delay Alert",

        "event_type 19":
            "Core Network Load Alert",

        "event_type 20":
            "Performance Degradation Alert",

        "event_type 21":
            "Routing Table Change",

        "event_type 22":
            "Route Instability Alert",

        "event_type 23":
            "Route Convergence Delay",

        "event_type 24":
            "BGP Peer Failure",

        "event_type 25":
            "OSPF Neighbor Failure",

        "event_type 26":
            "Route Advertisement Error",

        "event_type 27":
            "Route Policy Violation",

        "event_type 28":
            "Network Loop Detection",

        "event_type 29":
            "Gateway Failure",

        "event_type 30":
            "Traffic Rerouting Event",

        "event_type 31":
            "Hardware Health Warning",

        "event_type 32":
            "Device Failure Alert",

        "event_type 33":
            "Temperature Warning",

        "event_type 34":
            "Cooling Failure Alert",

        "event_type 35":
            "Power Instability Alert",

        "event_type 36":
            "Battery Failure Alert",

        "event_type 37":
            "Hardware Restart Event",

        "event_type 38":
            "Optical Signal Loss",

        "event_type 39":
            "Fiber Quality Warning",

        "event_type 40":
            "Physical Infrastructure Failure",

        "event_type 41":
            "Authentication Failure",

        "event_type 42":
            "Authorization Failure",

        "event_type 43":
            "DNS Resolution Failure",

        "event_type 44":
            "Application Service Failure",

        "event_type 45":
            "Database Service Failure",

        "event_type 46":
            "Configuration Error Alert",

        "event_type 47":
            "Firmware Error Alert",

        "event_type 48":
            "Security Threat Alert",

        "event_type 49":
            "Intrusion Detection Alert",

        "event_type 50":
            "DDoS Suspicion Alert",

        "event_type 51":
            "Anomaly Detection Alert",

        "event_type 52":
            "Critical Service Failure",

        "event_type 53":
            "System-Wide Outage Alert"
    }


    # ======================================================
    # FEATURE GROUP MAPPING
    # ======================================================

    def map_feature(
        feature
    ):

        match = re.search(
            r"(\d+)",
            str(feature)
        )

        if not match:

            return (
                "Unknown Feature Group"
            )

        value = int(
            match.group(1)
        )

        if 1 <= value <= 50:

            return (
                "Packet Quality Indicators"
            )

        if 51 <= value <= 100:

            return (
                "Traffic Indicators"
            )

        if 101 <= value <= 150:

            return (
                "Hardware Indicators"
            )

        if 151 <= value <= 200:

            return (
                "Power Indicators"
            )

        if 201 <= value <= 250:

            return (
                "Optical Indicators"
            )

        if 251 <= value <= 300:

            return (
                "Configuration Indicators"
            )

        if 301 <= value <= 386:

            return (
                "Security Indicators"
            )

        return (
            "Unknown Feature Group"
        )


    # ======================================================
    # RETURN SEMANTIC INCIDENT
    # ======================================================

    return {

        "severity":
            severity_map.get(
                ml_output["severity_type"],
                "Unknown Severity"
            ),

        "resource":
            resource_map.get(
                ml_output["resource_type"],
                "Unknown Resource"
            ),

        "events": [

            event_map.get(
                event,
                "Unknown Event"
            )

            for event in
            ml_output["event_types"]
        ],

        "feature_groups": [

            map_feature(feature)

            for feature in
            ml_output["log_features"]
        ],

        "predicted_fault_severity":
            ml_output[
                "predicted_fault_severity"
            ],

        "volume":
            ml_output["volume"],

        "raw_severity":
            ml_output["severity_type"],

        "raw_resource":
            ml_output["resource_type"],

        "raw_events":
            ml_output["event_types"],

        "raw_features":
            ml_output["log_features"]
    }


# ==========================================================
# HISTORICAL RCA EXTRACTION HELPERS
# ==========================================================

def extract_historical_candidates(pattern_context):
    """
    Extract historical RCA candidates from both formats used by
    the historical RAG collection:

        Resolved Root Cause:
        Successful Resolution:

    and:

        Historical Root Cause:
        Historical Solution:

    Returns complete candidates with an evidence score.
    """
    if not pattern_context:
        return []

    candidates = []

    # Split records using the raw-signature marker. This prevents
    # one historical paragraph from swallowing the next record.
    records = re.split(
        r"(?=RAW\s+INCIDENT\s+SIGNATURE\s*:)",
        pattern_context,
        flags=re.IGNORECASE
    )

    for record in records:
        if not record.strip():
            continue

        root_match = re.search(
            r"""
            (?:
                Resolved\s+Root\s+Cause
                |
                Historical\s+Root\s+Cause
            )
            \s*:\s*
            (.*?)
            (?=
                \n\s*
                (?:
                    Successful\s+Resolution
                    |
                    Historical\s+Solution
                )
                \s*:
                |
                \Z
            )
            """,
            record,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE
        )

        resolution_match = re.search(
            r"""
            (?:
                Successful\s+Resolution
                |
                Historical\s+Solution
            )
            \s*:\s*
            (.*?)
            (?=
                \n\s*
                RAW\s+INCIDENT\s+SIGNATURE
                |
                \Z
            )
            """,
            record,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE
        )

        if not root_match:
            continue

        root = root_match.group(1).strip().strip(".")
        resolution = (
            resolution_match.group(1).strip().strip(".")
            if resolution_match
            else ""
        )

        if not root:
            continue

        # Calculate evidence strength from the record itself.
        raw_signature = bool(
            re.search(
                r"RAW\s+INCIDENT\s+SIGNATURE",
                record,
                flags=re.IGNORECASE
            )
        )

        semantic_signature = bool(
            re.search(
                r"SEMANTIC\s+INCIDENT\s+SIGNATURE",
                record,
                flags=re.IGNORECASE
            )
        )

        exact_match_score = 0.90 if raw_signature and semantic_signature else 0.75

        candidates.append({
            "root_cause": root,
            "resolution": resolution,
            "confidence": exact_match_score,
            "evidence": (
                "Historical RAG exact incident-signature evidence "
                "with a previously successful resolution."
                if exact_match_score >= 0.90
                else
                "Historical RAG evidence with a previously "
                "successful resolution."
            )
        })

    # Deduplicate by root cause while preserving the strongest
    # evidence score and first useful resolution.
    deduped = {}

    for candidate in candidates:
        key = candidate["root_cause"].strip().lower()

        if key not in deduped:
            deduped[key] = candidate
            continue

        if (
            candidate["confidence"]
            > deduped[key]["confidence"]
        ):
            deduped[key]["confidence"] = candidate["confidence"]

        if (
            not deduped[key]["resolution"]
            and candidate["resolution"]
        ):
            deduped[key]["resolution"] = candidate["resolution"]

    return list(deduped.values())


# ==========================================================
# RCA VALIDATION / SANITIZATION
# ==========================================================

def _clean_inline_generation(text):
    """Normalize model output without changing semantic content."""
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```(?:text|plain|markdown)?", "", text, flags=re.I)
    text = text.replace("```", "")
    # Collapse excessive whitespace but preserve line boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_allowed_root_causes(knowledge_context, pattern_context, pattern_analysis):
    """Build the allow-list of RCA names supported by retrieved evidence."""
    allowed = []

    for source in (knowledge_context, pattern_context, pattern_analysis):
        if not source:
            continue

        matches = re.findall(
            r"(?:ROOT\s+CAUSE|Resolved\s+Root\s+Cause|Historical\s+Root\s+Cause)"
            r"\s*:\s*([^\n]+)",
            source,
            flags=re.I,
        )
        for value in matches:
            value = re.sub(r"\s+", " ", value).strip(" .:-")
            if value and len(value) < 160:
                allowed.append(value)

    # Deduplicate while preserving evidence order.
    result = []
    seen = set()
    for value in allowed:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _match_supported_root(root, allowed_roots):
    """Return the canonical retrieved root-cause name, or None."""
    if not root:
        return None

    cleaned = re.sub(r"\s+", " ", str(root)).strip(" .:-")
    key = cleaned.lower()

    for allowed in allowed_roots:
        akey = allowed.lower()
        if key == akey:
            return allowed

    # Permit small punctuation differences, but never accept a completely
    # unrelated model hallucination such as 'Routing instability'.
    for allowed in allowed_roots:
        akey = re.sub(r"[^a-z0-9]+", " ", allowed.lower()).strip()
        ckey = re.sub(r"[^a-z0-9]+", " ", key).strip()
        if ckey == akey:
            return allowed

    return None


def _historical_ranked_candidates(pattern_context, semantic_incident):
    """Return unique historical candidates, preferring exact raw-signature matches."""
    candidates = extract_historical_candidates(pattern_context)
    if not candidates:
        return []

    current_raw = {
        str(semantic_incident.get("raw_severity", "")).lower(),
        str(semantic_incident.get("raw_resource", "")).lower(),
        *[str(x).lower() for x in semantic_incident.get("raw_events", [])],
        *[str(x).lower() for x in semantic_incident.get("raw_features", [])],
    }

    # extract_historical_candidates already assigns 0.90 to raw+semantic
    # records. Keep that evidence score; it is a ranking score, not a
    # calibrated probability.
    candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return candidates


def _safe_fallback_candidates(pattern_context, knowledge_context, semantic_incident):
    """Construct three evidence-backed candidates without asking the LLM."""
    historical = _historical_ranked_candidates(pattern_context, semantic_incident)

    result = []
    seen = set()
    for item in historical:
        key = item["root_cause"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "root_cause": item["root_cause"],
            "resolution": item.get("resolution", ""),
            "confidence": item.get("confidence", 0.75),
            "evidence": item.get("evidence", "Historical RAG evidence."),
        })
        if len(result) == 3:
            return result

    # If fewer than three historical causes exist, supplement from the
    # knowledge RAG, but only with explicitly retrieved root causes.
    for root in re.findall(
        r"ROOT\s+CAUSE\s*:\s*([^\n]+)",
        knowledge_context or "",
        flags=re.I,
    ):
        root = root.strip().strip(".")
        key = root.lower()
        if not root or key in seen:
            continue

        # Find the corresponding remediation in the same knowledge block.
        resolution = ""
        block_match = re.search(
            rf"ROOT\s+CAUSE\s*:\s*{re.escape(root)}(.*?)(?=ROOT\s+CAUSE\s*:|\Z)",
            knowledge_context,
            flags=re.I | re.S,
        )
        if block_match:
            rem = re.search(
                r"REMEDIATION\s*:\s*(.*?)(?=ROOT\s+CAUSE\s*:|\Z)",
                block_match.group(1),
                flags=re.I | re.S,
            )
            if rem:
                resolution = re.sub(r"\s+", " ", rem.group(1)).strip()

        result.append({
            "root_cause": root,
            "resolution": resolution,
            "confidence": 0.70,
            "evidence": "Retrieved telecom domain-knowledge evidence.",
        })
        seen.add(key)
        if len(result) == 3:
            break

    return result


# ==========================================================
# PARSE PLAIN TEXT RCA
# ==========================================================

def parse_plain_text_rca(
    text,
    semantic_incident,
    pattern_analysis,
    pattern_context="",
    knowledge_context=""
):
    """
    Parse Ollama's plain-text RCA response.

    Supports:
      1. Explicit RCA_1/RCA_2/RCA_3 output.
      2. Natural-language output produced by telecom-copilot.
      3. Historical RAG fallback for remaining candidates.

    Confidence values such as 95%, 95, or 0.95 are normalized
    to the internal 0.0-1.0 representation.
    """

    if not text:
        raise RuntimeError(
            "Ollama returned an empty final RCA response."
        )

    text = str(text).strip()

    # Remove optional markdown fences.
    text = re.sub(
        r"```(?:text|plain|markdown)?",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = text.replace("```", "").strip()

    result = {
        "ranked_causes": [],
        "technical_summary": "",
        "risk_level": "UNKNOWN",
        "semantic_incident": semantic_incident,
        "pattern_analysis": pattern_analysis
    }

    normalized = _clean_inline_generation(text)

    allowed_roots = _extract_allowed_root_causes(
        knowledge_context,
        pattern_context,
        pattern_analysis
    )

    def parse_confidence(value):
        if value is None:
            return 0.0

        try:
            value = str(value).strip()

            if "%" in value:
                number = float(
                    value.replace("%", "").strip()
                )
                return max(
                    0.0,
                    min(1.0, number / 100.0)
                )

            number = float(value)

            if number > 1.0:
                number /= 100.0

            return max(
                0.0,
                min(1.0, number)
            )

        except (TypeError, ValueError):
            return 0.0

    # ======================================================
    # RISK LEVEL
    # ======================================================

    risk_match = re.search(
        r"(?:RISK[_ ]?LEVEL|RISK)\s*[:\-]\s*([^\n]+)",
        normalized,
        flags=re.IGNORECASE
    )

    if risk_match:
        result["risk_level"] = (
            risk_match.group(1).strip()
        )

    # ======================================================
    # TECHNICAL SUMMARY
    # ======================================================

    summary_match = re.search(
        r"""
        TECHNICAL[_ ]?SUMMARY
        \s*[:\-]\s*
        (.*?)
        (?=
            \n\s*RCA[_ ]?\d+
            |
            \n\s*ROOT[_ ]?CAUSE
            |
            \n\s*RESOLVED[_ ]ROOT[_ ]CAUSE
            |
            \Z
        )
        """,
        normalized,
        flags=re.IGNORECASE |
              re.DOTALL |
              re.VERBOSE
    )

    if summary_match:
        result["technical_summary"] = (
            summary_match.group(1).strip()
        )

    # ======================================================
    # METHOD 1: EXPLICIT RCA_1 / RCA_2 / RCA_3
    # ======================================================

    header_pattern = re.compile(
        r"""
        ^\s*
        RCA[_ \-]?(\d+)
        \s*[:.\-]?\s*
        """,
        flags=re.IGNORECASE |
              re.MULTILINE |
              re.VERBOSE
    )

    headers = list(
        header_pattern.finditer(normalized)
    )

    for index, header in enumerate(headers):

        block_start = header.end()

        block_end = (
            headers[index + 1].start()
            if index + 1 < len(headers)
            else len(normalized)
        )

        block = normalized[
            block_start:block_end
        ].strip()

        root_match = re.search(
            r"""
            ROOT[_ ]?CAUSE
            \s*[:\-]\s*
            (.*?)
            (?=
                \n\s*RESOLUTION
                |
                \n\s*CONFIDENCE
                |
                \n\s*EVIDENCE
                |
                \Z
            )
            """,
            block,
            flags=re.IGNORECASE |
                  re.DOTALL |
                  re.VERBOSE
        )

        resolution_match = re.search(
            r"""
            RESOLUTION
            \s*[:\-]\s*
            (.*?)
            (?=
                \n\s*CONFIDENCE
                |
                \n\s*EVIDENCE
                |
                \Z
            )
            """,
            block,
            flags=re.IGNORECASE |
                  re.DOTALL |
                  re.VERBOSE
        )

        confidence_match = re.search(
            r"""
            CONFIDENCE
            \s*[:\-]\s*
            ([0-9]+(?:\.[0-9]+)?%?)
            """,
            block,
            flags=re.IGNORECASE |
                  re.VERBOSE
        )

        evidence_match = re.search(
            r"""
            EVIDENCE
            \s*[:\-]\s*
            (.*)
            $
            """,
            block,
            flags=re.IGNORECASE |
                  re.DOTALL |
                  re.VERBOSE
        )

        root_cause = (
            root_match.group(1).strip()
            if root_match else ""
        )

        resolution = (
            resolution_match.group(1).strip()
            if resolution_match else ""
        )

        confidence = parse_confidence(
            confidence_match.group(1)
            if confidence_match else "0"
        )

        evidence = (
            evidence_match.group(1).strip()
            if evidence_match else ""
        )

        if root_cause:
            result["ranked_causes"].append({
                "rank": int(header.group(1)),
                "root_cause": root_cause.rstrip("."),
                "resolution": resolution.rstrip("."),
                "confidence": confidence,
                "evidence": evidence
            })

    # ======================================================
    # METHOD 2: NATURAL-LANGUAGE SINGLE RCA
    #
    # Handles the actual model output:
    #
    # Resolved root cause: Configuration Error.
    # Resolution: Rollback changes...
    # Confidence: 95%.
    # Evidence: ...
    # ======================================================

    if len(result["ranked_causes"]) == 0:

        root_match = re.search(
            r"""
            (?:
                ROOT[_ ]?CAUSE
                |
                RESOLVED[_ ]ROOT[_ ]CAUSE
            )
            \s*[:\-]\s*
            (.*?)
            (?=
                \n\s*
                (?:
                    DESCRIPTION
                    |
                    RESOLUTION
                    |
                    CONFIDENCE
                    |
                    EVIDENCE
                )
                \s*[:\-]
                |
                \Z
            )
            """,
            normalized,
            flags=re.IGNORECASE |
                  re.DOTALL |
                  re.VERBOSE
        )

        resolution_match = re.search(
            r"""
            RESOLUTION
            \s*[:\-]\s*
            (.*?)
            (?=
                \n\s*
                (?:
                    CONFIDENCE
                    |
                    EVIDENCE
                )
                \s*[:\-]
                |
                \Z
            )
            """,
            normalized,
            flags=re.IGNORECASE |
                  re.DOTALL |
                  re.VERBOSE
        )

        confidence_match = re.search(
            r"""
            CONFIDENCE
            \s*[:\-]\s*
            ([0-9]+(?:\.[0-9]+)?%?)
            """,
            normalized,
            flags=re.IGNORECASE |
                  re.VERBOSE
        )

        evidence_match = re.search(
            r"""
            EVIDENCE
            \s*[:\-]\s*
            (.*)
            $
            """,
            normalized,
            flags=re.IGNORECASE |
                  re.DOTALL |
                  re.VERBOSE
        )

        if root_match:

            result["ranked_causes"].append({
                "rank": 1,
                "root_cause": (
                    root_match.group(1)
                    .strip()
                    .rstrip(".")
                ),
                "resolution": (
                    resolution_match.group(1)
                    .strip()
                    .rstrip(".")
                    if resolution_match else ""
                ),
                "confidence": parse_confidence(
                    confidence_match.group(1)
                    if confidence_match else "0"
                ),
                "evidence": (
                    evidence_match.group(1).strip()
                    if evidence_match else ""
                )
            })

    # ======================================================
    # METHOD 3: HISTORICAL RAG FALLBACK
    # ======================================================

    if len(result["ranked_causes"]) < 3:

        print()
        print(
            "Ollama returned fewer than 3 RCA candidates."
        )
        print(
            "Recovering additional candidates "
            "from historical RAG evidence..."
        )

        historical_candidates = (
            extract_historical_candidates(
                pattern_context
            )
        )

        print(
            "Historical RCA candidates found:",
            len(historical_candidates)
        )

        existing_causes = {
            candidate["root_cause"].strip().lower()
            for candidate in result["ranked_causes"]
        }

        for historical in historical_candidates:

            root = historical["root_cause"]
            root_key = root.strip().lower()

            if root_key in existing_causes:
                continue

            result["ranked_causes"].append({
                "rank": len(result["ranked_causes"]) + 1,
                "root_cause": root,
                "resolution": historical["resolution"],
                "confidence": historical["confidence"],
                "evidence": historical["evidence"]
            })

            existing_causes.add(root_key)

            if len(result["ranked_causes"]) == 3:
                break

    # ======================================================
    # METHOD 4: PATTERN ANALYST FALLBACK
    # ======================================================

    if len(result["ranked_causes"]) < 3:

        pattern_root_matches = re.findall(
            r"""
            (?:
                Historical\s+root\s+cause
                |
                Resolved\s+root\s+cause
                |
                Root\s+cause
            )
            \s*[:\-]\s*
            ([^\n.]+)
            """,
            pattern_analysis,
            flags=re.IGNORECASE |
                  re.VERBOSE
        )

        existing_causes = {
            candidate["root_cause"].strip().lower()
            for candidate in result["ranked_causes"]
        }

        for root in pattern_root_matches:

            root = root.strip().rstrip(".")

            if not root or root.lower() in existing_causes:
                continue

            pattern_confidence_match = re.search(
                r"Confidence\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?%?)",
                pattern_analysis,
                flags=re.IGNORECASE
            )

            if pattern_confidence_match:
                raw_confidence = pattern_confidence_match.group(1)
                if "%" in raw_confidence:
                    pattern_confidence = (
                        float(raw_confidence.replace("%", "")) / 100.0
                    )
                else:
                    pattern_confidence = float(raw_confidence)
                    if pattern_confidence > 1.0:
                        pattern_confidence /= 100.0
                pattern_confidence = max(
                    0.0,
                    min(1.0, pattern_confidence)
                )
            else:
                pattern_confidence = 0.60

            resolution_match = re.search(
                r"(?:Successful\s+Resolution|Historical\s+Solution|Resolution)"
                r"\s*[:\-]\s*(.*?)(?=\n\s*(?:Confidence|Evidence)\s*[:\-]|\Z)",
                pattern_analysis,
                flags=re.IGNORECASE | re.DOTALL
            )

            pattern_resolution = (
                resolution_match.group(1).strip().strip(".")
                if resolution_match
                else "Review and remediate the identified condition."
            )

            result["ranked_causes"].append({
                "rank": len(result["ranked_causes"]) + 1,
                "root_cause": root,
                "resolution": pattern_resolution,
                "confidence": pattern_confidence,
                "evidence": (
                    "Identified from historical "
                    "pattern analysis."
                )
            })

            existing_causes.add(root.lower())

            if len(result["ranked_causes"]) == 3:
                break

    # ======================================================
    # VALIDATE MODEL OUTPUT AGAINST RETRIEVED EVIDENCE
    # ======================================================

    validated = []
    seen = set()

    for candidate in result["ranked_causes"]:
        canonical = _match_supported_root(
            candidate.get("root_cause", ""),
            allowed_roots
        )

        if not canonical:
            print(
                "Rejected unsupported LLM RCA:",
                candidate.get("root_cause", "")
            )
            continue

        key = canonical.lower()
        if key in seen:
            continue

        candidate["root_cause"] = canonical
        seen.add(key)
        validated.append(candidate)

    result["ranked_causes"] = validated

    # If the model hallucinated an unsupported root cause or returned fewer
    # than three candidates, deterministically recover from RAG.
    if len(result["ranked_causes"]) < 3:
        fallback = _safe_fallback_candidates(
            pattern_context,
            knowledge_context,
            semantic_incident
        )

        existing = {
            c["root_cause"].strip().lower()
            for c in result["ranked_causes"]
        }

        for item in fallback:
            if item["root_cause"].strip().lower() in existing:
                continue
            result["ranked_causes"].append({
                "rank": len(result["ranked_causes"]) + 1,
                **item
            })
            existing.add(item["root_cause"].strip().lower())
            if len(result["ranked_causes"]) == 3:
                break

    # ======================================================
    # FINAL VALIDATION
    # ======================================================

    if len(result["ranked_causes"]) != 3:

        print()
        print("=" * 100)
        print("RCA PARSER FAILED")
        print("=" * 100)

        print(
            "Expected 3 RCA candidates."
        )

        print(
            "Detected:",
            len(result["ranked_causes"])
        )

        print()
        print("RAW OLLAMA RESPONSE:")
        print(normalized)

        print()
        print("HISTORICAL PATTERN CONTEXT:")
        print(pattern_context)

        print("=" * 100)

        raise RuntimeError(
            "Unable to construct exactly 3 RCA candidates."
        )

    # ======================================================
    # RECOVER MISSING FIELDS FROM HISTORICAL RAG
    # ======================================================

    historical_candidates_for_recovery = (
        extract_historical_candidates(
            pattern_context
        )
    )

    historical_by_root = {
        item["root_cause"].strip().lower(): item
        for item in historical_candidates_for_recovery
    }

    for candidate in result["ranked_causes"]:

        root_key = (
            candidate.get("root_cause", "")
            .strip()
            .lower()
        )

        historical = historical_by_root.get(
            root_key
        )

        if historical:

            if not candidate.get("resolution"):
                candidate["resolution"] = (
                    historical["resolution"]
                )

            if not candidate.get("evidence"):
                candidate["evidence"] = (
                    historical["evidence"]
                )

            # If the LLM did not provide confidence, use
            # historical evidence strength rather than 0.0.
            if (
                not candidate.get("confidence")
                or float(candidate.get("confidence", 0.0)) == 0.0
            ):
                candidate["confidence"] = (
                    historical["confidence"]
                )

    # ======================================================
    # NORMALIZE
    # ======================================================

    for index, candidate in enumerate(
        result["ranked_causes"],
        start=1
    ):

        candidate["rank"] = index

        candidate["root_cause"] = str(
            candidate.get("root_cause", "")
        ).strip()

        candidate["resolution"] = str(
            candidate.get("resolution", "")
        ).strip()

        candidate["evidence"] = str(
            candidate.get("evidence", "")
        ).strip()

        try:
            candidate["confidence"] = float(
                candidate.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            candidate["confidence"] = 0.0

        candidate["confidence"] = max(
            0.0,
            min(1.0, candidate["confidence"])
        )

    if not result["risk_level"]:
        result["risk_level"] = "UNKNOWN"

    if not result["technical_summary"]:
        result["technical_summary"] = (
            "The incident was analyzed using "
            "current telecom telemetry, domain knowledge, "
            "and historical incident patterns."
        )

    # ======================================================
    # PRINT FINAL PARSED RCA
    # ======================================================

    print()
    print("=" * 100)
    print("PARSED FINAL RCA")
    print("=" * 100)

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print()
    print("Technical Summary:")
    print(result["technical_summary"])

    print()

    for candidate in result["ranked_causes"]:

        print(
            f"RCA #{candidate['rank']}"
        )

        print(
            "Root Cause :",
            candidate["root_cause"]
        )

        print(
            "Resolution :",
            candidate["resolution"]
        )

        print(
            "Confidence :",
            candidate["confidence"]
        )

        print(
            "Evidence   :",
            candidate["evidence"]
        )

        print()

    print("=" * 100)

    return result


# ==========================================================
# GENERATE AGENTIC RCA
# ==========================================================

def generate_rca_agentic(
    ml_output
):

    # ======================================================
    # STEP 1 — BUILD RAW QUERY
    # ======================================================

    query = build_query(
        ml_output
    )


    # ======================================================
    # STEP 2 — BUILD SEMANTIC INCIDENT
    # ======================================================

    semantic_incident = (
        build_semantic_incident(
            ml_output
        )
    )


    # ======================================================
    # STEP 3 — SEMANTIC QUERY
    # ======================================================

    semantic_query = f"""
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


    # ======================================================
    # STEP 4 — KNOWLEDGE RETRIEVAL
    # ======================================================

    try:

        knowledge_docs = (
            knowledge_retriever.invoke(
                semantic_query
            )
        )

    except Exception as exc:

        print()
        print(
            "Knowledge Retrieval Error:",
            exc
        )

        knowledge_docs = []


    knowledge_context = "\n\n".join(

        doc.page_content

        for doc in knowledge_docs
    )


    # ======================================================
    # STEP 5 — HISTORICAL PATTERN RETRIEVAL
    # ======================================================

    pattern_query = f"""
RAW INCIDENT:
{query}

SEMANTIC INCIDENT:
{semantic_query}
""".strip()


    try:

        pattern_docs = (
            pattern_retriever.invoke(
                pattern_query
            )
        )

    except Exception as exc:

        print()
        print(
            "Pattern Retrieval Error:",
            exc
        )

        pattern_docs = []


    pattern_context = "\n\n".join(

        doc.page_content

        for doc in pattern_docs
    )


    # ======================================================
    # SHOW MAPPED INCIDENT
    # ======================================================

    print()
    print("=" * 100)
    print(
        "MAPPED INCIDENT"
    )
    print("=" * 100)

    print(
        json.dumps(
            semantic_incident,
            indent=2
        )
    )


    # ======================================================
    # SHOW KNOWLEDGE
    # ======================================================

    print()
    print("=" * 100)
    print(
        "RETRIEVED TELECOM KNOWLEDGE"
    )
    print("=" * 100)

    print(
        knowledge_context
    )


    # ======================================================
    # SHOW HISTORICAL PATTERNS
    # ======================================================

    print()
    print("=" * 100)
    print(
        "RETRIEVED HISTORICAL PATTERNS"
    )
    print("=" * 100)

    print(
        pattern_context
    )


    # ======================================================
    # STEP 6 — PATTERN ANALYST AGENT
    # ======================================================

    pattern_prompt = f"""
You are a Telecom Historical Incident Pattern Analyst.

Analyze ONLY the retrieved historical incidents.

CURRENT INCIDENT:
{semantic_query}

HISTORICAL INCIDENTS:
{pattern_context}

Identify the strongest historical evidence.

Return:

1. Most relevant historical root cause
2. Successful resolution
3. Why the historical incident matches
4. Confidence

Do not invent information.

If there is an exact incident-signature match,
explicitly mention it.

Do not produce generic telecom advice.

Return plain text only.
"""


    try:

        check_ollama_connection()

        pattern_analysis = call_ollama(
            pattern_prompt
        )

    except Exception as exc:

        print()
        print(
            "Pattern Agent Error:",
            exc
        )

        pattern_analysis = ""


    # ======================================================
    # SHOW PATTERN ANALYSIS
    # ======================================================

    print()
    print("=" * 100)
    print(
        "PATTERN ANALYST AGENT"
    )
    print("=" * 100)

    print(
        pattern_analysis
    )


    # ======================================================
    # STEP 7 — FINAL RCA AGENT
    # ======================================================

    allowed_root_text = "\n".join(
        f"- {root}" for root in _extract_allowed_root_causes(
            knowledge_context, pattern_context, pattern_analysis
        )
    )

    final_prompt = f"""
You are a telecom RCA ranking agent.

IMPORTANT: You MUST select a root cause from the SUPPORTED ROOT CAUSES list.
Never invent a new root cause.

CURRENT INCIDENT:
{semantic_query}

SUPPORTED ROOT CAUSES:
{allowed_root_text}

TELECOM KNOWLEDGE:
{knowledge_context}

HISTORICAL EVIDENCE:
{pattern_context}

PATTERN ANALYSIS:
{pattern_analysis}

TASK:
Choose the strongest evidence-backed root cause.
Prefer an exact historical incident-signature match.
Use the successful resolution from the matching evidence.

Return exactly these four lines and nothing else:
Resolved root cause: <one supported root cause>
Resolution: <one specific resolution>
Confidence: <0 to 100>
Evidence: <short evidence>
"""

    # ======================================================
    # CALL OLLAMA
    # ======================================================

    try:

        check_ollama_connection()

        final_content = call_ollama(
            final_prompt
        )

    except Exception as exc:

        print()
        print(
            "Final RCA Ollama Error:",
            exc
        )

        raise RuntimeError(
            "Final RCA generation failed."
        ) from exc


    # ======================================================
    # SHOW RAW OLLAMA RESPONSE
    # ======================================================

    print()
    print("=" * 100)
    print(
        "RAW FINAL OLLAMA RESPONSE"
    )
    print("=" * 100)

    print()

    print(
        final_content
    )

    print()

    print("=" * 100)


    # ======================================================
    # HANDLE EMPTY RESPONSE
    # ======================================================

    if not final_content:

        raise RuntimeError(
            "Ollama returned an empty final RCA response."
        )


    # ======================================================
    # PARSE PLAIN TEXT
    # ======================================================

    result = parse_plain_text_rca(

        final_content,

        semantic_incident,

        pattern_analysis,

        pattern_context,

        knowledge_context
    )


    # ======================================================
    # SHOW PARSED RCA
    # ======================================================

    print()
    print("=" * 100)
    print(
        "PARSED FINAL RCA"
    )
    print("=" * 100)


    print()
    print(
        "Risk Level:"
    )

    print(
        result["risk_level"]
    )


    print()
    print(
        "Technical Summary:"
    )

    print(
        result["technical_summary"]
    )


    print()
    print(
        "Ranked RCA Candidates:"
    )


    for candidate in (
        result["ranked_causes"]
    ):

        print()
        print(
            f"#{candidate['rank']}"
        )

        print(
            "Root Cause  :",
            candidate["root_cause"]
        )

        print(
            "Confidence  :",
            candidate["confidence"]
        )

        print(
            "Resolution  :",
            candidate["resolution"]
        )

        print(
            "Evidence    :",
            candidate["evidence"]
        )


    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {

        "ranked_causes":
            result["ranked_causes"],

        "technical_summary":
            result["technical_summary"],

        "risk_level":
            result["risk_level"],

        "semantic_incident":
            semantic_incident,

        "pattern_analysis":
            pattern_analysis
    }


# ==========================================================
# BACKWARD COMPATIBILITY
# ==========================================================

def generate_rca(
    ml_output
):

    return generate_rca_agentic(
        ml_output
    )