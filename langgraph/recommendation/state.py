from __future__ import annotations

from typing import Any, TypedDict
class RecommendationState(TypedDict, total=False):
    # Input (from Root Cause subgraph)
    root_cause: str
    root_cause_confidence: float
    root_cause_analysis: dict[str, Any]
    root_cause_metadata: dict[str, Any]

    # Light context passthrough
    location: str | None
    severity_type: str | None
    resource_type: str | None
    fault_severity: int | float | None

    # Produced by the graph
    solution_category: str
    recommended_solution: str
    action_items: list[str]
    priority: str
    confidence: float
    metadata: dict[str, Any]

    # Control / error handling
    is_valid: bool
    error: str | None