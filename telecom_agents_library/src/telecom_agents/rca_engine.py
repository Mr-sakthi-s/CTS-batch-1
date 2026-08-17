import json
import os
import re

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MAPPING_KB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "mapping_KB.txt"
)

VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "vector_db"
)


# ==========================================================
# EMBEDDING MODEL
# ==========================================================

EMBEDDING_MODEL = (
    r"C:\Users\sakthi murugan\.cache\huggingface\hub"
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
    search_kwargs={"k": 5}
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
    search_kwargs={"k": 5}
)


# ==========================================================
# LLM
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv("TELECOM_AGENTS_LLM_MODEL", "gemini-3.5-flash-lite")
)


# ==========================================================
# LOAD MAPPING KB
# ==========================================================

def load_mapping_kb():

    if not os.path.exists(
        MAPPING_KB_PATH
    ):
        raise FileNotFoundError(
            f"Mapping KB not found:\n"
            f"{MAPPING_KB_PATH}"
        )

    with open(
        MAPPING_KB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    severity_map = {}
    resource_map = {}
    event_map = {}
    feature_groups = []

    section = None

    for line in lines:

        # ----------------------------------------------
        # Detect sections
        # ----------------------------------------------

        if "SEVERITY TYPE MAPPING" in line:
            section = "severity"
            continue

        if "RESOURCE TYPE MAPPING" in line:
            section = "resource"
            continue

        if "EVENT TYPE MAPPING" in line:
            section = "event"
            continue

        if "LOG FEATURE GROUPS" in line:
            section = "feature"
            continue

        # ----------------------------------------------
        # Severity
        # ----------------------------------------------

        if section == "severity":

            match = re.match(
                r"^(severity_type\s+\d+)$",
                line
            )

            if match:
                current_key = match.group(1)
                continue

            if (
                "current_key" in locals()
                and line.startswith("Category:")
            ):

                severity_map[
                    current_key
                ] = line.replace(
                    "Category:",
                    ""
                ).strip()

                continue

        # ----------------------------------------------
        # Resource
        # ----------------------------------------------

        if section == "resource":

            match = re.match(
                r"^(resource_type\s+\d+)$",
                line
            )

            if match:
                current_key = match.group(1)
                continue

            if (
                "current_key" in locals()
                and line.startswith("Category:")
            ):

                resource_map[
                    current_key
                ] = line.replace(
                    "Category:",
                    ""
                ).strip()

                continue

        # ----------------------------------------------
        # Event
        # ----------------------------------------------

        if section == "event":

            match = re.match(
                r"^(event_type\s+\d+)$",
                line
            )

            if match:
                current_key = match.group(1)
                continue

            if (
                "current_key" in locals()
                and not line.startswith("=")
                and not line.startswith("NOTE:")
            ):

                event_map[
                    current_key
                ] = line

                continue

        # ----------------------------------------------
        # Feature groups
        # ----------------------------------------------

        if section == "feature":

            match = re.match(
                r"^(feature\s+\d+)(?:-(\d+))?$",
                line
            )

            if match:

                start = int(
                    match.group(1).split()[-1]
                )

                end = (
                    int(match.group(2))
                    if match.group(2)
                    else start
                )

                # Next non-empty line is the group
                feature_groups.append(
                    {
                        "start": start,
                        "end": end,
                        "group": None
                    }
                )

                continue

            if (
                feature_groups
                and feature_groups[-1]["group"] is None
            ):

                feature_groups[-1][
                    "group"
                ] = line

    return {
        "severity": severity_map,
        "resource": resource_map,
        "event": event_map,
        "feature": feature_groups
    }


# ==========================================================
# GLOBAL MAPPING
# ==========================================================

MAPPING = load_mapping_kb()


# ==========================================================
# FEATURE LOOKUP
# ==========================================================

def map_feature(
    feature_name
):

    match = re.search(
        r"feature\s+(\d+)",
        str(feature_name)
    )

    if not match:
        return "Unknown Feature Group"

    feature_number = int(
        match.group(1)
    )

    for item in MAPPING["feature"]:

        if (
            item["start"]
            <= feature_number
            <= item["end"]
        ):

            return item["group"]

    return "Unknown Feature Group"


# ==========================================================
# MAP ML OUTPUT TO SEMANTIC INCIDENT
# ==========================================================

def map_ml_output(
    ml_output
):

    severity_id = (
        ml_output[
            "severity_type"
        ]
    )

    resource_id = (
        ml_output[
            "resource_type"
        ]
    )

    event_ids = (
        ml_output[
            "event_types"
        ]
    )

    feature_ids = (
        ml_output[
            "log_features"
        ]
    )

    severity_meaning = MAPPING[
        "severity"
    ].get(
        severity_id,
        "Unknown Severity"
    )

    resource_meaning = MAPPING[
        "resource"
    ].get(
        resource_id,
        "Unknown Resource"
    )

    event_meanings = [
        MAPPING[
            "event"
        ].get(
            event_id,
            "Unknown Event"
        )
        for event_id in event_ids
    ]

    feature_meanings = [
        map_feature(
            feature_id
        )
        for feature_id in feature_ids
    ]

    semantic_incident = {
        "predicted_fault_severity":
            ml_output[
                "predicted_fault_severity"
            ],

        "severity_id":
            severity_id,

        "severity":
            severity_meaning,

        "resource_id":
            resource_id,

        "resource":
            resource_meaning,

        "event_ids":
            event_ids,

        "events":
            event_meanings,

        "feature_ids":
            feature_ids,

        "feature_groups":
            feature_meanings,

        "volume":
            ml_output[
                "volume"
            ]
    }

    return semantic_incident


# ==========================================================
# BUILD RAW + SEMANTIC SEARCH QUERY
# ==========================================================

def build_search_query(
    ml_output,
    semantic_incident
):

    raw_part = f"""
Raw Incident Identifiers:
Severity Type: {ml_output['severity_type']}
Resource Type: {ml_output['resource_type']}
Event Types: {", ".join(ml_output['event_types'])}
Log Features: {", ".join(ml_output['log_features'])}
Predicted Fault Severity: {ml_output['predicted_fault_severity']}
Volume: {ml_output['volume']}
"""

    semantic_part = f"""
Semantic Telecom Interpretation:

Severity:
{semantic_incident['severity']}

Resource:
{semantic_incident['resource']}

Events:
{", ".join(semantic_incident['events'])}

Feature Groups:
{", ".join(semantic_incident['feature_groups'])}

Volume:
{semantic_incident['volume']}
"""

    return (
        raw_part
        + "\n"
        + semantic_part
    )


# ==========================================================
# RESPONSE EXTRACTION
# ==========================================================

def extract_response_text(
    response
):

    if isinstance(
        response.content,
        list
    ):

        content = ""

        for block in response.content:

            if isinstance(
                block,
                dict
            ):

                content += block.get(
                    "text",
                    ""
                )

            else:

                content += str(block)

        return content

    return str(
        response.content
    )


# ==========================================================
# GENERATE RCA
# ==========================================================

def generate_rca(
    ml_output
):

    # ------------------------------------------------------
    # MAP ANONYMIZED ML OUTPUT
    # ------------------------------------------------------

    semantic_incident = map_ml_output(
        ml_output
    )

    # ------------------------------------------------------
    # Build semantic retrieval query
    # ------------------------------------------------------

    query = build_search_query(
        ml_output,
        semantic_incident
    )

    # ------------------------------------------------------
    # Knowledge retrieval
    # ------------------------------------------------------

    knowledge_docs = (
        knowledge_retriever.invoke(
            query
        )
    )

    knowledge_context = (
        "\n\n".join(
            doc.page_content
            for doc in knowledge_docs
        )
    )

    # ------------------------------------------------------
    # Pattern retrieval
    # ------------------------------------------------------

    pattern_docs = (
        pattern_retriever.invoke(
            query
        )
    )

    pattern_context = (
        "\n\n".join(
            doc.page_content
            for doc in pattern_docs
        )
    )

    # ======================================================
    # DISPLAY MAPPING
    # ======================================================

    print("\n")
    print("=" * 100)
    print("ML OUTPUT → MAPPING KB")
    print("=" * 100)

    print(
        f"Severity: "
        f"{ml_output['severity_type']}"
        f" → "
        f"{semantic_incident['severity']}"
    )

    print(
        f"Resource: "
        f"{ml_output['resource_type']}"
        f" → "
        f"{semantic_incident['resource']}"
    )

    print(
        "\nEvents:"
    )

    for raw, meaning in zip(
        semantic_incident[
            "event_ids"
        ],
        semantic_incident[
            "events"
        ]
    ):

        print(
            f"{raw} → {meaning}"
        )

    print(
        "\nFeatures:"
    )

    for raw, meaning in zip(
        semantic_incident[
            "feature_ids"
        ],
        semantic_incident[
            "feature_groups"
        ]
    ):

        print(
            f"{raw} → {meaning}"
        )

    # ======================================================
    # RETRIEVED KNOWLEDGE
    # ======================================================

    print("\n")
    print("=" * 100)
    print("RETRIEVED TELECOM KNOWLEDGE")
    print("=" * 100)

    print(
        knowledge_context
    )

    # ======================================================
    # RETRIEVED PATTERNS
    # ======================================================

    print("\n")
    print("=" * 100)
    print("RETRIEVED HISTORICAL PATTERNS")
    print("=" * 100)

    print(
        pattern_context
    )

    # ======================================================
    # LLM PROMPT
    # ======================================================

    prompt = f"""
You are an expert Telecom Root Cause Analysis Agent.

The ML model produces anonymized identifiers.
A synthetic telecom mapping layer translates those
identifiers into human-readable telecom concepts.

IMPORTANT:

Do not claim that the raw ML identifiers inherently
have telecom meaning.

The mapping layer provides their semantic interpretation.

==================================================
RAW ML OUTPUT
==================================================

{json.dumps(
    ml_output,
    indent=2
)}

==================================================
MAPPED TELECOM INTERPRETATION
==================================================

{json.dumps(
    semantic_incident,
    indent=2
)}

==================================================
RETRIEVED TELECOM KNOWLEDGE
==================================================

{knowledge_context}

==================================================
RETRIEVED HISTORICAL RESOLUTIONS
==================================================

{pattern_context}

==================================================
TASK
==================================================

Generate the three strongest root-cause hypotheses.

Use BOTH:

1. Mapped telecom domain knowledge.
2. Historical successful resolutions.

IMPORTANT:

- Prefer an exact or near-exact historical successful
  resolution when the incident signature strongly matches.
- Treat historical resolution memory as evidence,
  not absolute truth.
- Use the telecom knowledge base to generate and
  validate alternative hypotheses.
- Each hypothesis must have a different root cause.
- Each hypothesis must have exactly one technician-
  actionable resolution.
- Do NOT invent evidence.
- Do NOT claim an event directly proves a root cause
  unless the retrieved evidence supports that conclusion.

For each hypothesis include an evidence field explaining
why it was selected.

Return ONLY valid JSON.

Required format:

{{
    "ranked_causes": [
        {{
            "rank": 1,
            "root_cause": "",
            "resolution": "",
            "confidence": 0.0,
            "evidence": ""
        }},
        {{
            "rank": 2,
            "root_cause": "",
            "resolution": "",
            "confidence": 0.0,
            "evidence": ""
        }},
        {{
            "rank": 3,
            "root_cause": "",
            "resolution": "",
            "confidence": 0.0,
            "evidence": ""
        }}
    ],
    "technical_summary": "",
    "risk_level": ""
}}
"""

    # ======================================================
    # GENERATE
    # ======================================================

    response = llm.invoke(
        prompt
    )

    content = (
        extract_response_text(
            response
        )
    )

    content = re.sub(
        r"```json|```",
        "",
        content
    ).strip()

    # ======================================================
    # PARSE
    # ======================================================

    try:

        result = json.loads(
            content
        )

        candidates = result.get(
            "ranked_causes",
            []
        )

        if len(candidates) < 3:

            raise ValueError(
                "RCA did not return 3 candidates"
            )

        candidates = candidates[:3]

        # Normalize output
        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            candidate["rank"] = index

            candidate["root_cause"] = str(
                candidate.get(
                    "root_cause",
                    "Unknown"
                )
            ).strip()

            candidate["resolution"] = str(
                candidate.get(
                    "resolution",
                    "Manual investigation required"
                )
            ).strip()

            candidate["evidence"] = str(
                candidate.get(
                    "evidence",
                    "No evidence supplied."
                )
            ).strip()

            try:

                candidate["confidence"] = float(
                    candidate.get(
                        "confidence",
                        0.0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                candidate["confidence"] = 0.0

        # --------------------------------------------------
        # Return both raw mapping and RCA
        # --------------------------------------------------

        return {
            "ranked_causes": candidates,

            "technical_summary": str(
                result.get(
                    "technical_summary",
                    ""
                )
            ),

            "risk_level": str(
                result.get(
                    "risk_level",
                    "UNKNOWN"
                )
            ),

            "semantic_incident":
                semantic_incident
        }

    except Exception as exc:

        print("\n")
        print("=" * 80)
        print("RCA JSON ERROR")
        print("=" * 80)

        print(exc)

        print(
            "\nRaw response:"
        )

        print(
            content
        )

        return {
            "ranked_causes": [
                {
                    "rank": 1,
                    "root_cause":
                        "Unknown",
                    "resolution":
                        "Manual investigation required",
                    "confidence":
                        0.0,
                    "evidence":
                        "RCA generation failed."
                },
                {
                    "rank": 2,
                    "root_cause":
                        "Unknown",
                    "resolution":
                        "Inspect affected resource",
                    "confidence":
                        0.0,
                    "evidence":
                        "RCA generation failed."
                },
                {
                    "rank": 3,
                    "root_cause":
                        "Unknown",
                    "resolution":
                        "Escalate to NOC",
                    "confidence":
                        0.0,
                    "evidence":
                        "RCA generation failed."
                }
            ],

            "technical_summary":
                "RCA generation failed.",

            "risk_level":
                "UNKNOWN",

            "semantic_incident":
                semantic_incident
        }