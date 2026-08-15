from __future__ import annotations

import logging
from typing import Any

from recommendation.state import RecommendationState

logger = logging.getLogger("recommendation")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


_VALID_PRIORITIES = ("low", "medium", "high", "critical")


def validate_input(state: RecommendationState) -> dict[str, Any]:
    
    logger.info("validate_input: checking input state")

    root_cause = state.get("root_cause")
    if not isinstance(root_cause, str) or not root_cause.strip():
        msg = "root_cause is missing, empty, or not a string."
        logger.warning("validate_input: %s", msg)
        return {"is_valid": False, "error": msg}

    root_cause_confidence = state.get("root_cause_confidence")
    if not isinstance(root_cause_confidence, (int, float)):
        msg = "root_cause_confidence is missing or not numeric."
        logger.warning("validate_input: %s", msg)
        return {"is_valid": False, "error": msg}

    if not (0.0 <= float(root_cause_confidence) <= 1.0):
        msg = f"root_cause_confidence {root_cause_confidence} outside [0.0, 1.0]."
        logger.warning("validate_input: %s", msg)
        return {"is_valid": False, "error": msg}

    logger.info("validate_input: input is valid")
    return {"is_valid": True, "error": None}


def analyze_root_cause(state: RecommendationState) -> dict[str, Any]:
    
    logger.info("analyze_root_cause: classifying root cause")

    root_cause_text = (state.get("root_cause") or "").lower()
    root_cause_metadata = state.get("root_cause_metadata") or {}
    rule_fired = (root_cause_metadata.get("evidence") or {}).get("rule_fired")

    rule_to_category = {
        "no_prediction_available": "insufficient_data",
        "no_fault_detected": "no_action_needed",
        "moderate_fault_event_driven": "transient_event",
        "moderate_fault_general": "moderate_general",
        "severe_fault_event_and_log_driven": "systemic_issue",
        "severe_fault_low_log_diversity": "concentrated_issue",
        "severe_fault_general": "severe_general",
        "unrecognized_severity_value": "unknown",
    }

    if rule_fired in rule_to_category:
        solution_category = rule_to_category[rule_fired]
    elif "no significant fault" in root_cause_text:
        solution_category = "no_action_needed"
    elif "no fault severity prediction" in root_cause_text:
        solution_category = "insufficient_data"
    elif "systemic issue" in root_cause_text:
        solution_category = "systemic_issue"
    elif "severe" in root_cause_text:
        solution_category = "severe_general"
    elif "moderate" in root_cause_text:
        solution_category = "moderate_general"
    else:
        solution_category = "unknown"

    logger.info("analyze_root_cause: solution_category=%s", solution_category)
    return {"solution_category": solution_category}


def _rule_based_recommendation_engine(
    state: RecommendationState, solution_category: str
) -> tuple[str, list[str], str, float, dict[str, Any]]:
   
    location = state.get("location", "the affected location")
    resource_type = state.get("resource_type", "the affected resource")

    evidence: dict[str, Any] = {"solution_category": solution_category}

    catalog: dict[str, tuple[str, list[str], str, float]] = {
        "insufficient_data": (
            f"Hold off on remediation for {location} until a fault "
            f"severity prediction is available; recommend monitoring "
            f"{resource_type} in the meantime.",
            [
                "Re-run root cause analysis once a prediction is available.",
                f"Add {location} to the monitoring watchlist.",
                "Collect additional log samples if volume is low.",
            ],
            "low",
            0.2,
        ),
        "no_action_needed": (
            f"No remediation required for {location}; metrics for "
            f"{resource_type} are within expected ranges.",
            [
                "No action required.",
                "Continue routine monitoring.",
            ],
            "low",
            0.6,
        ),
        "transient_event": (
            f"Investigate transient/intermittent failures on "
            f"{resource_type} at {location}; likely self-resolving but "
            f"should be tracked to confirm it doesn't recur.",
            [
                f"Review recent event logs for {resource_type}.",
                "Set up alerting for repeat occurrences within 24h.",
                "No immediate escalation required unless it recurs.",
            ],
            "medium",
            0.5,
        ),
        "moderate_general": (
            f"Perform a targeted review of {resource_type} at "
            f"{location} to identify the dominant contributing factor.",
            [
                f"Pull detailed logs for {resource_type} over the last 24-48h.",
                "Compare against historical baseline for this location.",
                "Escalate to on-call engineer if pattern persists.",
            ],
            "medium",
            0.45,
        ),
        "systemic_issue": (
            f"Treat this as a systemic issue on {resource_type} at "
            f"{location}: both event volume and log volume are "
            f"elevated, suggesting a broad underlying problem rather "
            f"than an isolated incident.",
            [
                f"Escalate {resource_type} at {location} to the infrastructure team immediately.",
                "Check for recent deployments, config changes, or capacity issues.",
                "Consider temporary load shedding or failover if available.",
                "Schedule a post-incident review once resolved.",
            ],
            "critical",
            0.75,
        ),
        "concentrated_issue": (
            f"Investigate a small, repeating set of log features on "
            f"{resource_type} at {location}; the issue appears "
            f"concentrated rather than broad-based.",
            [
                "Identify the specific repeating log feature(s) driving the fault.",
                f"Check recent changes specific to that subsystem of {resource_type}.",
                "Apply a targeted fix rather than a broad remediation.",
            ],
            "high",
            0.65,
        ),
        "severe_general": (
            f"High-impact issue detected on {resource_type} at "
            f"{location}; detailed investigation required before a "
            f"specific fix can be recommended.",
            [
                f"Assign an engineer to investigate {resource_type} at {location} within the hour.",
                "Gather full log-level detail for the affected window.",
                "Notify stakeholders of a potential high-impact issue.",
            ],
            "high",
            0.5,
        ),
        "unknown": (
            f"Root cause classification was inconclusive for "
            f"{resource_type} at {location}; manual review recommended.",
            [
                "Manually review the root cause output and raw features.",
                "Do not apply automated remediation until reviewed.",
            ],
            "medium",
            0.1,
        ),
    }

    recommended_solution, action_items, priority, confidence = catalog.get(
        solution_category, catalog["unknown"]
    )
    evidence["matched_category"] = (
        solution_category if solution_category in catalog else "unknown (fallback)"
    )

    return recommended_solution, list(action_items), priority, confidence, evidence


def generate_recommendation(state: RecommendationState) -> dict[str, Any]:

    logger.info("generate_recommendation: running rule-based recommendation engine")

    solution_category = state.get("solution_category", "unknown")
    (
        recommended_solution,
        action_items,
        priority,
        confidence,
        evidence,
    ) = _rule_based_recommendation_engine(state, solution_category)

    existing_metadata = dict(state.get("metadata") or {})
    existing_metadata["recommendation_engine"] = "rule_based_v1"
    existing_metadata["evidence"] = evidence

    logger.info(
        "generate_recommendation: category=%s priority=%s confidence=%.2f",
        solution_category,
        priority,
        confidence,
    )

    return {
        "recommended_solution": recommended_solution,
        "action_items": action_items,
        "priority": priority,
        "confidence": confidence,
        "metadata": existing_metadata,
    }


def validate_recommendation(state: RecommendationState) -> dict[str, Any]:
   
    logger.info("validate_recommendation: validating final recommendation output")

    recommended_solution = state.get("recommended_solution")
    action_items = state.get("action_items")
    priority = state.get("priority")
    confidence = state.get("confidence")
    metadata = dict(state.get("metadata") or {})

    problems: list[str] = []

    if not isinstance(recommended_solution, str) or not recommended_solution.strip():
        problems.append("recommended_solution is empty or not a string")
        recommended_solution = "No recommendation could be generated from available data."

    if not isinstance(action_items, list) or not action_items:
        problems.append("action_items is empty or not a list")
        action_items = ["Manually review this case; no automated action items available."]

    if priority not in _VALID_PRIORITIES:
        problems.append(f"priority '{priority}' not in {_VALID_PRIORITIES}")
        priority = "medium"

    if not isinstance(confidence, (int, float)):
        problems.append("confidence is missing or not numeric")
        confidence = 0.0
    elif not (0.0 <= float(confidence) <= 1.0):
        problems.append(f"confidence {confidence} outside [0.0, 1.0]")
        confidence = max(0.0, min(1.0, float(confidence)))

    metadata["validated"] = True
    metadata["validation_problems"] = problems

    if problems:
        logger.warning("validate_recommendation: issues found: %s", problems)
    else:
        logger.info("validate_recommendation: recommendation passed validation")

    return {
        "recommended_solution": recommended_solution,
        "action_items": action_items,
        "priority": priority,
        "confidence": float(confidence),
        "metadata": metadata,
        "is_valid": not problems,
        "error": None if not problems else "; ".join(problems),
    }