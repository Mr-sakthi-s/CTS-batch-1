from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from recommendation.nodes import (
    analyze_root_cause,
    generate_recommendation,
    validate_input,
    validate_recommendation,
)
from recommendation.state import RecommendationState

logger = logging.getLogger("recommendation")


def _route_after_validation(state: RecommendationState) -> Literal["valid", "invalid"]:

    if state.get("is_valid"):
        return "valid"
    return "invalid"


def build_recommendation_graph():

    graph_builder = StateGraph(RecommendationState)

    graph_builder.add_node("validate_input", validate_input)
    graph_builder.add_node("analyze_root_cause", analyze_root_cause)
    graph_builder.add_node("generate_recommendation", generate_recommendation)
    graph_builder.add_node("validate_recommendation", validate_recommendation)

    graph_builder.add_edge(START, "validate_input")

    graph_builder.add_conditional_edges(
        "validate_input",
        _route_after_validation,
        {
            "valid": "analyze_root_cause",
            "invalid": END,
        },
    )

    graph_builder.add_edge("analyze_root_cause", "generate_recommendation")
    graph_builder.add_edge("generate_recommendation", "validate_recommendation")
    graph_builder.add_edge("validate_recommendation", END)

    compiled = graph_builder.compile()
    logger.info("Recommendation graph compiled successfully.")
    return compiled


# Ready-to-use compiled graph instance for convenient importing, e.g.:
#     from recommendation.workflow import recommendation_graph
#     result = recommendation_graph.invoke(input_data)
recommendation_graph = build_recommendation_graph()


if __name__ == "__main__":
    sample_input = {
        "root_cause": "Severe fault at location 118 on resource 'resource_type 2' (severity type 'severity_type 2'). Feature signals indicate a high-impact issue; detailed log-level investigation is recommended.",
        "root_cause_confidence": 0.75,
        "root_cause_analysis": {
            "location": "location 118",
            "severity_type": "severity_type 2",
            "resource_type": "resource_type 2",
            "fault_severity_provided": True,
            "event_diversity_ratio": 1.0,
            "log_diversity_ratio": 1.0,
            "total_log_volume": 38,
            "mean_log_volume": 19.0,
            "volume_spread": 0,
            "high_event_volume": False,
            "high_log_volume": False,
        },
        "root_cause_metadata": {
            "root_cause_engine": "rule_based_v1",
            "evidence": {"rule_fired": "severe_fault_general"},
        },
        "location": "location 118",
        "severity_type": "severity_type 2",
        "resource_type": "resource_type 2",
        "fault_severity": 2,
    }
    result = recommendation_graph.invoke(sample_input)

    print("\n========== RECOMMENDATION RESULT ==========\n")
    print(f"is_valid: {result.get('is_valid')}\n")
    print("recommended_solution:")
    print(result.get("recommended_solution"))
    print("\naction_items:")
    print(result.get("action_items"))
    print("\npriority:")
    print(result.get("priority"))
    print("\nconfidence:")
    print(result.get("confidence"))
    print("\nmetadata:")
    print(result.get("metadata"))
    print("\nerror:")
    print(result.get("error"))
