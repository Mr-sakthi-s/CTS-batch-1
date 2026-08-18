# agent/__init__.py
"""
Telecom fault-resolution agent package (LangGraph).

Public API:

    from agent import build_graph, run_pipeline, AgentState

    graph = build_graph()
    result = run_pipeline(ml_output, fault)
"""

from .graph import build_graph, graph, run_pipeline
from .state import AgentState, RCACandidate, Ticket

__all__ = [
    "AgentState",
    "RCACandidate",
    "Ticket",
    "build_graph",
    "graph",
    "run_pipeline",
]
