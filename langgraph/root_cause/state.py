from __future__ import annotations

from typing import TypedDict


class RootCauseState(TypedDict, total=False):

    severity_type: int | str
    resource_type: int | str
    event_type: int | str
    feature: int | str
    volume: int | str

    query: str
    retrieved_context: list[str]
    retrieved_incidents: list[dict[str, str]]

    root_cause: str
    root_cause_reason: str
    root_cause_explanation: str
    root_cause_characteristics: list[str]

    detailed_solution: list[str]
    solution: list[str]
    validation: list[str]

    final_report: str

    is_valid: bool
    error: str | None