# agent/state.py
"""
Shared state schema for the telecom fault-resolution LangGraph.

Every node reads from and writes partial updates to this single
``AgentState``. LangGraph merges each node's returned dict into
the running state.
"""

from typing import Any, Dict, List, Optional, TypedDict


# ==========================================================
# RCA CANDIDATE
# ==========================================================

class RCACandidate(TypedDict):
    """One ranked root-cause hypothesis produced by the RCA engine."""

    rank: int
    root_cause: str
    resolution: str
    confidence: float          # normalized 0.0 - 1.0
    evidence: str


# ==========================================================
# TICKET
# ==========================================================

class Ticket(TypedDict, total=False):
    """Lifecycle record for a single incident."""

    ticket_id: str
    status: str                # OPEN | IN_PROGRESS | CLOSED | ESCALATED
    attempt: int               # index of the RCA hypothesis being tried
    ranked_causes: List[RCACandidate]


# ==========================================================
# GRAPH STATE
# ==========================================================

class AgentState(TypedDict, total=False):
    # ------------------------------------------------------
    # INPUTS (provided when the graph is invoked)
    # ------------------------------------------------------
    ml_output: Dict[str, Any]
    # Raw ML pipeline output:
    #   severity_type, resource_type, event_types,
    #   log_features, predicted_fault_severity, volume

    fault: Dict[str, Any]
    # Fault record for dispatch:
    #   id, location, fault_severity, resource_type

    feedback_queue: List[bool]
    # Optional simulation hook: pre-seeded fixed/not-fixed
    # answers consumed one per verification attempt. In
    # production, leave empty and pause the graph at the
    # "verify" node instead (interrupt_before=["verify"]).

    # ------------------------------------------------------
    # RCA ENGINE OUTPUT
    # ------------------------------------------------------
    semantic_incident: Dict[str, Any]
    knowledge_context: str
    pattern_context: str
    pattern_analysis: str
    ranked_causes: List[RCACandidate]
    technical_summary: str
    risk_level: str

    # ------------------------------------------------------
    # RESOLUTION LOOP
    # ------------------------------------------------------
    ticket: Ticket
    attempt: int
    current_candidate: Optional[RCACandidate]
    fixed: bool                # did the last attempted resolution work?

    # ------------------------------------------------------
    # DISPATCH
    # ------------------------------------------------------
    dispatch_result: Dict[str, Any]

    # ------------------------------------------------------
    # OUTCOME
    # ------------------------------------------------------
    status: str                # IN_PROGRESS | RETRY | CLOSED | ESCALATE
    escalation: Optional[Dict[str, Any]]
    memory_saved: bool
