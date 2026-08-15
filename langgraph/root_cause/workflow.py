from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from root_cause.state import RootCauseState
from root_cause.nodes import (
    retrieve_context,
    analyze_root_cause,
    generate_detailed_solution,
)

logger = logging.getLogger("root_cause")


def build_root_cause_graph():

    graph_builder = StateGraph(RootCauseState)

    graph_builder.add_node("retrieve_context", retrieve_context)
    graph_builder.add_node("analyze_root_cause", analyze_root_cause)
    graph_builder.add_node("generate_detailed_solution", generate_detailed_solution)

    graph_builder.add_edge(START, "retrieve_context")
    graph_builder.add_edge("retrieve_context", "analyze_root_cause")
    graph_builder.add_edge("analyze_root_cause", "generate_detailed_solution")
    graph_builder.add_edge("generate_detailed_solution", END)

    compiled = graph_builder.compile()
    logger.info("Root cause graph compiled successfully.")
    return compiled


def process_live_fault_record(record: dict[str, Any]) -> dict[str, Any]:
    """Entry point for incoming live Wi-Fi fault records.

    The live input does not need to be hardcoded. Any caller that receives a
    network payload can pass the dictionary directly into this function or as an
    argument to root_cause_graph.invoke(...). The graph will then perform the
    retrieval and LLM analysis steps for that single incoming record.
    """
    if not isinstance(record, dict):
        return {"error": "Live input must be a dictionary-like record."}

    live_state = {
        "severity_type": record.get("severity_type"),
        "resource_type": record.get("resource_type"),
        "event_type": record.get("event_type"),
        "feature": record.get("feature"),
        "volume": record.get("volume"),
    }
    return root_cause_graph.invoke(live_state)


# Ready-to-use compiled graph instance for convenient importing, e.g.:
#     from root_cause.workflow import root_cause_graph
#     result = root_cause_graph.invoke(input_data)
root_cause_graph = build_root_cause_graph()


if __name__ == "__main__":
    sample_input = {
        "severity_type": 5,
        "resource_type": 8,
        "event_type": 11,
        "feature": 68,
        "volume": 250,
    }

    result = process_live_fault_record(sample_input)
    print(result.get("final_report"))