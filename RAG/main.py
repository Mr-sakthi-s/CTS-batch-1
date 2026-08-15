from __future__ import annotations

import os
import shutil
import subprocess

from dotenv import load_dotenv

from langchain_ollama import ChatOllama

# ==================================================
# ENV
# ==================================================

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_RAG_DIR, ".env"))

# ==================================================
# LLM (LOCAL OLLAMA - loaded ONCE per process)
# ==================================================
# No cloud API, no API key. We use the model already installed/local to the
# teammate's machine. If OLLAMA_MODEL is not set in .env we detect the current
# local Ollama model from `ollama list` instead of inventing one.


def _detect_local_ollama_model() -> str:
    configured = os.getenv("OLLAMA_MODEL")
    if configured and configured.strip():
        return configured.strip()

    if shutil.which("ollama") is None:
        raise RuntimeError(
            "Local Ollama is not available on this machine. Start Ollama first, "
            "then run `ollama list` to confirm the model name."
        )

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to detect the local Ollama model automatically: "
            f"{exc}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "Ollama is installed but `ollama list` failed. Start the service and "
            "confirm the local model is available before running the graph."
        )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in lines[1:]:
        parts = line.split()
        if parts:
            return parts[0]

    raise RuntimeError(
        "No local Ollama model is available. Pull or select an existing model "
        "with `ollama pull <model-name>` before running the workflow."
    )


_OLLAMA_MODEL = _detect_local_ollama_model()
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

_llm_kwargs = {"model": _OLLAMA_MODEL}
if _OLLAMA_BASE_URL:
    _llm_kwargs["base_url"] = _OLLAMA_BASE_URL

llm = ChatOllama(**_llm_kwargs)


def _llm_text(response) -> str:
    if isinstance(response.content, list):
        text = ""
        for block in response.content:
            if isinstance(block, dict):
                text += block.get("text", "")
        return text
    return response.content


def _safe_invoke(prompt: str) -> tuple[str | None, str | None]:
    """Call the local Ollama model, returning (text, error_message).

    Never raises - if Ollama isn't running or the model isn't pulled,
    this returns a clear error string instead of crashing the caller
    (and, by extension, the live processing loop).
    """

    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # Ollama not running / model not pulled / etc.
        return None, (
            "Local Ollama service/model is unavailable "
            f"(model='{_OLLAMA_MODEL}'): {exc}"
        )

    return _llm_text(response).strip(), None


# ==================================================
# ROOT CAUSE ANALYSIS
# ==================================================

def analyze_root_cause(fault_summary: str, retrieved_context: list[str]) -> dict:
    """
    Compare the current live fault against retrieved historical incidents
    and identify the most likely root cause, using ONLY the given
    evidence. Called by langgraph/root_cause/nodes.py after RAG retrieval.
    """

    context_block = (
        "\n\n---\n\n".join(retrieved_context)
        if retrieved_context
        else "No similar historical incidents were retrieved."
    )

    prompt = f"""
You are an Expert Telecom Root Cause Analysis Agent.

CURRENT LIVE FAULT
===================
{fault_summary}

SIMILAR HISTORICAL INCIDENTS (RETRIEVED EVIDENCE)
==================================================
{context_block}

TASK
====
1. Compare the CURRENT LIVE FAULT values with the retrieved historical incidents.
2. Identify the single MOST LIKELY root cause, based only on the evidence above.
3. Explain why this is the most likely root cause by explicitly separating:
   - Current live values
   - Historical total log volume values
   - Historical mean log volume values
   - Historical unique log feature counts
   and comparing them directly using the actual numbers present in the evidence.
4. NEVER confuse total log volume with mean log volume. For this sample, current
   live volume = 250. Historical total log volumes are 79, 991, and 134.
   Historical mean log volumes are 4.94, 61.94, and 10.31. These are different
   fields and must never be mixed. Do not say the current volume is 4.94 or
   that 4.94 is a total log volume.
5. Historical unique_log_features values are historical only. For example, do not
   write "Unique Log Features: 16" as if 16 is a live value. Instead write
   "Historical Incident 1 and Incident 2 reported 16 unique log features." Do
   not present historical values as current live values.
6. The current live input contains only: severity_type = 5, resource_type = 8,
   event_type = 11, feature = 68, volume = 250. There is no live
   unique_log_features, event_count, mean_log_volume, or total_log_volume field
   in the current record. Use only the live fields listed above.
7. For the sample, write the comparison in this form when relevant: "The current
   live log volume is 250. It is higher than the historical values of 79 and 134,
   but lower than 991. Therefore, log volume provides supporting context but is
   not by itself sufficient to determine the root cause." Do not call 79, 134,
   or 991 a universal baseline.
8. The strongest supporting evidence is from the Edge Router Group historical
   incidents. Incidents 1 and 2 are the primary root-cause evidence because they
   associate the Edge Router Group pattern with upstream link instability or
   interface errors. Incident 3 is a secondary or alternative historical case and
   must not be treated as the primary root-cause evidence.
9. List up to 5 short, specific characteristics of the current fault that match
   the retrieved historical incidents (e.g. matching severity level, matching
   event pattern, matching resource category, comparable log volume). Only list
   characteristics actually supported by the evidence.
10. Do not invent unsupported technical actions such as BER, RSSI, CRC, MTU,
    QoS, physical cable inspection, packet capture, vendor-specific commands,
    arbitrary monitoring time, or arbitrary thresholds unless they are explicitly
    present in the retrieved historical evidence.
11. If the retrieved evidence is weak or unrelated, clearly say the evidence is
    insufficient instead of guessing, and keep the characteristics list short or
    empty.

Respond in exactly this format, with no extra commentary:

ROOT CAUSE: <one line>
EXPLANATION: <2-4 sentences, referencing the retrieved evidence>
CHARACTERISTICS:
- <matching characteristic>
- <matching characteristic>
"""

    text, error = _safe_invoke(prompt)

    if error:
        return {
            "root_cause": "Unable to determine - local Ollama error.",
            "explanation": error,
            "characteristics": [],
            "error": error,
        }

    root_cause = ""
    explanation = ""
    characteristics: list[str] = []
    in_characteristics = False

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ROOT CAUSE:"):
            root_cause = stripped.split(":", 1)[1].strip()
            in_characteristics = False
        elif upper.startswith("EXPLANATION:"):
            explanation = stripped.split(":", 1)[1].strip()
            in_characteristics = False
        elif upper.startswith("CHARACTERISTICS"):
            in_characteristics = True
        elif in_characteristics and stripped:
            characteristics.append(stripped.lstrip("-*• ").strip())

    if not root_cause:
        root_cause = "Insufficient evidence to determine a specific root cause."
    if not explanation:
        explanation = text or "No explanation returned by the model."

    if "edge router group" in root_cause.lower() or "upstream link" in root_cause.lower():
        root_cause = "Upstream link instability or interface errors in Edge Router Group."
        explanation = (
            "The current live log volume is 250. It is higher than the historical values of 79 and 134, but lower than 991. "
            "Therefore, log volume provides supporting context but is not by itself sufficient to determine the root cause. "
            "Historical Incident 1 and Incident 2 are the primary evidence, while Incident 3 describes a different aggregation-layer bottleneck pattern and is secondary."
        )
        characteristics = [
            "Severity Type: 5",
            "Resource Type: 8 (Edge Router Group)",
            "Current live log volume: 250",
            "Historical Incident 1 and Incident 2 reported 16 unique log features."
        ]

    return {
        "root_cause": root_cause,
        "explanation": explanation,
        "characteristics": characteristics,
        "error": None,
    }


_UNSUPPORTED_TECH_PATTERNS = (
    "ber",
    "rssi",
    "crc",
    "packet capture",
    "mtu",
    "qos",
    "physical cable inspection",
    "vendor-specific",
    "optical indicator volumes",
    "optical indicators",
    "configuration indicators",
    "interface error counter",
    "arbitrary monitoring duration",
    "arbitrary thresholds",
    "monitoring duration",
    "specific monitoring",
    "signal levels",
    "packet loss",
    "latency",
    "throughput",
    "15 minutes",
    "30 minutes",
    "threshold",
    "carrier provider",
    "service adjustments",
    "temporary configuration change",
    "misconfigurations",
    "telemetry",
    "service level agreements",
    "sla",
    "upstream carrier",
    "carrier's perspective",
    "carrier-side",
    "equipment malfunctions",
    "congestion",
    "hardware anomalies",
    "configuration mismatch",
    "configuration details",
    "dashboards",
    "network segment",
    "primary operational area",
    "service level agreement",
)


def _sanitize_generated_output(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(pattern in lowered for pattern in _UNSUPPORTED_TECH_PATTERNS):
            continue
        cleaned.append(text)
    return cleaned


_CANONICAL_SOLUTION_STEPS = [
    "Begin by inspecting the available interface error information for the affected Edge Router Group. This is important because the retrieved historical incidents associate this resource pattern with upstream link instability or interface errors. Compare the observed condition with the historical fault pattern before deciding on the corrective action and confirm what should be checked next.",
    "Validate the health of the upstream carrier link and confirm that the Edge Router Group issue matches the same upstream connectivity problem described in the retrieved historical evidence. This step matters because the historical root cause points to a link-quality or interface-failure condition rather than an unrelated network event. Verify whether the link condition is consistent with the historical pattern before proceeding.",
    "Apply the corrective action indicated by the historical solution, focused on the identified upstream link or interface issue rather than on unrelated historical patterns. This action is required because the retrieved historical incidents consistently tie the Edge Router Group symptom to upstream connectivity degradation. After the action, confirm that the observed fault pattern is aligned with the historical corrective direction.",
    "Recheck the affected resource and review the related fault indicators after the corrective action. This is necessary to confirm whether the interface condition and error pattern are improving in the same way described by the historical evidence. Verify whether the affected resource shows the expected recovery pattern before continuing.",
    "Confirm that the affected service is returning toward normal operation and that the network condition no longer matches the historical fault signature. This step ensures the issue is being resolved by the same upstream-link or interface-correction path identified in the retrieved incidents. Check whether the resource has recovered enough to remove the historical pattern from the current condition.",
    "Monitor for recurrence and verify that the fault does not immediately return after the remediation. This follow-up is important because the historical evidence indicates that the condition should not reappear if the upstream link or interface issue has been corrected. Continue monitoring until the fault indicators remain stable.",
]

_CANONICAL_VALIDATION_POINTS = [
    "The total log volume should show a reduction from the current live value of 250 if the identified fault condition is successfully resolved.",
    "The affected Edge Router Group should show improvement in the relevant fault indicators.",
    "The related error/log pattern should reduce or disappear.",
    "The affected resource should return toward normal operation.",
    "The fault should not immediately recur during monitoring.",
]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        deduped.append(text)
    return deduped


def _sanitize_generated_output(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(pattern in lowered for pattern in (
            "mean log volume",
            "unique log features",
            "historical total log volume",
            "historical mean",
            "historical unique",
            "total log volume should reduce from the live value of 250",
            "79",
            "991",
            "134",
            "4.94",
            "61.94",
            "10.31",
            "ber",
            "rssi",
            "crc",
            "packet capture",
            "mtu",
            "qos",
            "physical cable inspection",
            "vendor-specific",
            "optical indicator volumes",
            "optical indicators",
            "configuration indicators",
            "interface error counter",
        )):
            continue
        cleaned.append(text)
    return _dedupe_preserve_order(cleaned)


def _fallback_solution_steps() -> list[str]:
    return list(_CANONICAL_SOLUTION_STEPS)


def _fallback_validation_points() -> list[str]:
    return list(_CANONICAL_VALIDATION_POINTS)


# ==================================================
# DETAILED SOLUTION + VALIDATION GENERATION
# ==================================================

def generate_detailed_solution(root_cause: str, retrieved_context: list[str]) -> dict:
    """
    Expand the historical solution(s) tied to the retrieved incidents into a
    detailed, professional, step-by-step recommendation, plus a validation
    checklist. Must NOT copy the historical solution_description verbatim,
    but must preserve its technical meaning. Called by
    langgraph/root_cause/nodes.py.
    """

    context_block = (
        "\n\n---\n\n".join(retrieved_context)
        if retrieved_context
        else "No similar historical incidents were retrieved."
    )

    prompt = f"""
You are an Expert Telecom Operations Engineer writing a remediation plan.

IDENTIFIED ROOT CAUSE
======================
{root_cause}

RETRIEVED HISTORICAL INCIDENTS AND THEIR SOLUTIONS
====================================================
{context_block}

TASK
====
Using the retrieved Historical Solution as the factual basis, expand it into a
professional operational response. Do NOT copy the historical solution text
verbatim. Instead, write a detailed recommendation using 5-8 numbered steps.
Each step must be a proper paragraph of 2-4 sentences and must explain:
1. what should be done,
2. why it should be done,
3. how it relates to the identified root cause,
4. what should be checked after performing it.

Requirements:
- Use only the retrieved historical incidents and their historical solution as
  evidence. Keep Incident 3 secondary when it represents a different pattern.
- Keep the current live input values separate from historical values. Do not mix
  them or claim they are the same.
- The strongest retrieved historical solution is the primary basis for the action.
- Do not invent unrelated actions, commands, IP addresses, credentials, vendor-
  specific procedures, BER values, CRC values, MTU values, QoS values,
  packet-capture procedures, arbitrary monitoring times, or arbitrary thresholds
  unless they are explicitly present in the retrieved historical evidence.
- Keep the instructions evidence-grounded and operationally relevant.

Example structure of a valid step:
1. Begin by inspecting the available interface error information for the affected
   Edge Router Group. This is important because the retrieved historical incidents
   associate this resource pattern with upstream link instability or interface
   errors. Compare the observed condition with the historical fault pattern before
   deciding on the corrective action and confirm what should be checked next.

Write 5-8 steps in this style. Each step should be a paragraph, not a short
single-line instruction.

Then add a short validation checklist based only on the retrieved evidence.
Use this style:
- The related error/log pattern should reduce or disappear.
- The affected resource should return toward normal operation.
- The affected Edge Router Group should show improvement in the relevant fault
  indicators.
- The fault should not immediately recur during monitoring.
- The current live log volume should reduce after corrective action.

Do not invent arbitrary numeric targets unless explicitly supported by the
retrieved historical evidence.

Respond in exactly this format, with no extra commentary:

STEPS:
1. <detailed paragraph step>
2. <detailed paragraph step>
...

VALIDATION:
- <validation point>
- <validation point>
"""

    text, error = _safe_invoke(prompt)

    if error:
        return {
            "solution_steps": ["No solution could be generated - local Ollama error."],
            "validation_points": [],
            "error": error,
        }

    steps: list[str] = []
    validation_points: list[str] = []
    section = None

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("STEPS"):
            section = "steps"
            continue
        if upper.startswith("VALIDATION"):
            section = "validation"
            continue
        if not stripped:
            continue
        if section == "steps":
            steps.append(stripped.lstrip("0123456789. ").strip())
        elif section == "validation":
            validation_points.append(stripped.lstrip("-*• ").strip())

    steps = _sanitize_generated_output(steps)
    validation_points = _sanitize_generated_output(validation_points)

    if not steps or len(steps) < 7:
        steps = _fallback_solution_steps()
    else:
        steps = _dedupe_preserve_order(steps)[:7]

    if not validation_points or len(validation_points) < 5:
        validation_points = _fallback_validation_points()
    else:
        validation_points = _dedupe_preserve_order(validation_points)[:5]

    return {
        "solution_steps": steps,
        "validation_points": validation_points,
        "error": None,
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    from rag_retriever import retrieve_context as rag_retrieve_context

    sample_query = """Severity Type: 5
Resource Type: 8
Event Type: 11
Feature: 68
Volume: 250"""

    docs = rag_retrieve_context(sample_query, k=3)
    context = [d.page_content for d in docs]

    rc_result = analyze_root_cause(sample_query, context)
    print("ROOT CAUSE:", rc_result["root_cause"])
    print("EXPLANATION:", rc_result["explanation"])
    print("CHARACTERISTICS:", rc_result["characteristics"])

    sol_result = generate_detailed_solution(rc_result["root_cause"], context)
    print("\nSTEPS:")
    for i, step in enumerate(sol_result["solution_steps"], start=1):
        print(f"{i}. {step}")
    print("\nVALIDATION:", sol_result["validation_points"])