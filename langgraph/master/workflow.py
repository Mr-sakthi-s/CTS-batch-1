from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from root_cause.workflow import root_cause_graph
from recommendation.workflow import recommendation_graph

logger = logging.getLogger("master")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class MasterState(TypedDict, total=False):

    # Raw input
    input_data: dict[str, Any]
    fault_severity: int | float | None
    severity_type: str | None
    resource_type: str | None
    location: str | None
    event_count: int | float | None
    unique_event_count: int | float | None
    log_feature_count: int | float | None
    unique_log_features: int | float | None
    total_log_volume: int | float | None
    mean_log_volume: int | float | None
    max_log_volume: int | float | None
    min_log_volume: int | float | None

    # Root Cause subgraph output (prefixed)
    root_cause: str
    root_cause_confidence: float
    root_cause_analysis: dict[str, Any]
    root_cause_metadata: dict[str, Any]
    root_cause_is_valid: bool
    root_cause_error: str | None

    # Recommendation subgraph output (prefixed)
    recommended_solution: str
    action_items: list[str]
    priority: str
    recommendation_confidence: float
    recommendation_metadata: dict[str, Any]
    recommendation_is_valid: bool
    recommendation_error: str | None

    # Overall pipeline status
    is_valid: bool
    error: str | None


_ROOT_CAUSE_INPUT_KEYS = (
    "input_data",
    "fault_severity",
    "severity_type",
    "resource_type",
    "location",
    "event_count",
    "unique_event_count",
    "log_feature_count",
    "unique_log_features",
    "total_log_volume",
    "mean_log_volume",
    "max_log_volume",
    "min_log_volume",
)


def run_root_cause_analysis(state: MasterState) -> dict[str, Any]:

    logger.info("run_root_cause_analysis: invoking Root Cause subgraph")

    rc_input = _extract_root_cause_input(state)
    rc_result = root_cause_graph.invoke(rc_input)

    is_valid = bool(rc_result.get("is_valid"))
    logger.info("run_root_cause_analysis: subgraph is_valid=%s", is_valid)

    original_context = dict(state)
    if "input_data" in original_context and isinstance(original_context["input_data"], dict):
        original_context.update(original_context["input_data"])

    return {
        "root_cause": rc_result.get("root_cause"),
        "root_cause_confidence": rc_result.get("confidence"),
        "root_cause_analysis": rc_result.get("analysis"),
        "root_cause_metadata": rc_result.get("metadata"),
        "root_cause_is_valid": is_valid,
        "root_cause_error": rc_result.get("error"),
        "location": original_context.get("location"),
        "severity_type": original_context.get("severity_type"),
        "resource_type": original_context.get("resource_type"),
        "fault_severity": original_context.get("fault_severity"),
    }


def run_recommendation(state: MasterState) -> dict[str, Any]:

    logger.info("run_recommendation: invoking Recommendation subgraph")

    rec_input = {
        "root_cause": state.get("root_cause"),
        "root_cause_confidence": state.get("root_cause_confidence"),
        "root_cause_analysis": state.get("root_cause_analysis"),
        "root_cause_metadata": state.get("root_cause_metadata"),
        "location": state.get("location"),
        "severity_type": state.get("severity_type"),
        "resource_type": state.get("resource_type"),
        "fault_severity": state.get("fault_severity"),
    }
    rec_result = recommendation_graph.invoke(rec_input)

    is_valid = bool(rec_result.get("is_valid"))
    logger.info("run_recommendation: subgraph is_valid=%s", is_valid)

    return {
        "recommended_solution": rec_result.get("recommended_solution"),
        "action_items": rec_result.get("action_items"),
        "priority": rec_result.get("priority"),
        "recommendation_confidence": rec_result.get("confidence"),
        "recommendation_metadata": rec_result.get("metadata"),
        "recommendation_is_valid": is_valid,
        "recommendation_error": rec_result.get("error"),
        "is_valid": is_valid,
        "error": rec_result.get("error"),
    }


def _route_after_root_cause(state: MasterState) -> Literal["continue", "stop"]:
  
    if state.get("root_cause_is_valid"):
        return "continue"
    return "stop"


def _finalize_root_cause_failure(state: MasterState) -> dict[str, Any]:

    return {
        "is_valid": False,
        "error": state.get("root_cause_error") or "Root cause analysis failed.",
    }


def _extract_root_cause_input(state: MasterState) -> dict[str, Any]:
    if "input_data" in state and isinstance(state["input_data"], dict):
        payload = dict(state["input_data"])
        for key in _ROOT_CAUSE_INPUT_KEYS:
            if key != "input_data" and key not in state and key in payload:
                state[key] = payload[key]
        return payload
    return {key: state[key] for key in _ROOT_CAUSE_INPUT_KEYS if key in state}


def build_master_graph():

    graph_builder = StateGraph(MasterState)

    graph_builder.add_node("run_root_cause_analysis", run_root_cause_analysis)
    graph_builder.add_node("run_recommendation", run_recommendation)
    graph_builder.add_node("root_cause_failure", _finalize_root_cause_failure)

    graph_builder.add_edge(START, "run_root_cause_analysis")

    graph_builder.add_conditional_edges(
        "run_root_cause_analysis",
        _route_after_root_cause,
        {
            "continue": "run_recommendation",
            "stop": "root_cause_failure",
        },
    )

    graph_builder.add_edge("run_recommendation", END)
    graph_builder.add_edge("root_cause_failure", END)

    compiled = graph_builder.compile()
    logger.info("Master graph compiled successfully.")
    return compiled


# Ready-to-use compiled graph instance, e.g.:
#     from master.workflow import master_graph
#     result = master_graph.invoke(input_data)
master_graph = build_master_graph()


if __name__ == "__main__":
    sample_input = {
        "input_data": {
            "id": 14121,
            "location": "location 118",
            "fault_severity": 1,
            "severity_type": "severity_type 2",
            "resource_type": "resource_type 2",
            "event_count": 2,
            "unique_event_count": 2,
            "log_feature_count": 2,
            "unique_log_features": 2,
            "total_log_volume": 38,
            "mean_log_volume": 19.0,
            "max_log_volume": 19,
            "min_log_volume": 19,
        }
    }
    result = master_graph.invoke(sample_input)

    print("\n========== FINAL RESULT ==========\n")
    print(f"Root Cause: {result.get('root_cause')}")
    print(f"Root Cause Confidence: {result.get('root_cause_confidence')}")
    print(f"Recommended Solution: {result.get('recommended_solution')}")
    print(f"Action Items: {result.get('action_items')}")
    print(f"Priority: {result.get('priority')}")
    print(f"Recommendation Confidence: {result.get('recommendation_confidence')}")
    print(f"Overall Status: {result.get('is_valid')}")
    print(f"Error: {result.get('error')}")
