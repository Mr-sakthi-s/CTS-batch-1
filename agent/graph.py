# agent/graph.py
"""
LangGraph wiring for the telecom fault-resolution pipeline.

Flow:

                 START
                   |
                   v
              +---------+
              |   rca   |   agentic RCA -> 3 ranked hypotheses
              +---------+
                   |
                   v
              +----------+
              | dispatch |  technician + spare part
              +----------+
                |      \
   (no technician)      \
                |        v
                |   +---------+
                |   | verify  |  did the fix work?  <----+
                |   +---------+                          |
                |        |                               |
                |        v                               |
                |   +----------+        RETRY            |
                |   | feedback |  -----------------------+
                |   +----------+   (next RCA hypothesis)
                |     |      |
                |  CLOSED  ESCALATE
                |     |      |
                |     v      v
                | +--------+ +------------+
                | | memory | | escalation |
                | +--------+ +------------+
                |     |            |
                v     v            v
                      END
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    dispatch_node,
    escalation_node,
    feedback_node,
    memory_node,
    rca_node,
    verify_node,
)
from .state import AgentState


# ==========================================================
# ROUTERS (conditional edges)
# ==========================================================

def route_after_dispatch(state: AgentState) -> str:
    """No technician available anywhere -> escalate immediately."""
    if state.get("status") == "ESCALATE":
        return "escalation"
    return "verify"


def route_after_feedback(state: AgentState) -> str:
    """
    CLOSED   -> save to self-learning memory
    RETRY    -> technician tries the next ranked hypothesis
    ESCALATE -> hand off to the human NOC team
    """
    status = state.get("status")

    if status == "CLOSED":
        return "memory"
    if status == "RETRY":
        return "verify"
    return "escalation"


# ==========================================================
# GRAPH BUILDER
# ==========================================================

def build_graph(checkpointer=None, human_in_loop: bool = False):
    """
    Build and compile the fault-resolution graph.

    Args:
        checkpointer:
            Optional LangGraph checkpointer (e.g. MemorySaver)
            for persistence / pausing.
        human_in_loop:
            When True, the graph pauses before every "verify"
            step so a technician (or monitoring system) can
            report whether the fix worked. Resume with
            ``graph.invoke({"fixed": True}, config)``.
            Requires a checkpointer.
    """
    workflow = StateGraph(AgentState)

    # -- NODES ---------------------------------------------
    workflow.add_node("rca", rca_node)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("feedback", feedback_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("escalation", escalation_node)

    # -- EDGES ---------------------------------------------
    workflow.add_edge(START, "rca")
    workflow.add_edge("rca", "dispatch")

    workflow.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {
            "verify": "verify",
            "escalation": "escalation",
        },
    )

    workflow.add_edge("verify", "feedback")

    workflow.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "memory": "memory",
            "verify": "verify",       # retry loop
            "escalation": "escalation",
        },
    )

    workflow.add_edge("memory", END)
    workflow.add_edge("escalation", END)

    # -- COMPILE -------------------------------------------
    compile_kwargs = {}

    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    if human_in_loop:
        compile_kwargs["interrupt_before"] = ["verify"]

    return workflow.compile(**compile_kwargs)


# Module-level compiled graph (also usable by `langgraph dev`).
graph = build_graph()


# ==========================================================
# CONVENIENCE RUNNER
# ==========================================================

def run_pipeline(
    ml_output: dict,
    fault: dict,
    feedback_queue: list | None = None,
) -> AgentState:
    """
    Run one incident end-to-end.

    Args:
        ml_output:
            Raw ML pipeline output (severity_type, resource_type,
            event_types, log_features, predicted_fault_severity,
            volume).
        fault:
            Fault record (id, location, fault_severity,
            resource_type).
        feedback_queue:
            Optional list of booleans simulating whether each
            resolution attempt worked, e.g. [False, False, True].
    """
    initial_state: AgentState = {
        "ml_output": ml_output,
        "fault": fault,
        "feedback_queue": feedback_queue or [],
    }

    # recursion_limit guards the verify<->feedback retry loop.
    return graph.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )


# ==========================================================
# DEMO
# ==========================================================

if __name__ == "__main__":

    sample_ml_output = {
        "severity_type": "severity_type 4",
        "resource_type": "resource_type 2",
        "event_types": ["event_type 24", "event_type 22"],
        "log_features": ["feature 265", "feature 271"],
        "predicted_fault_severity": 2,
        "volume": 14,
    }

    sample_fault = {
        "id": 118,
        "location": "location 118",
        "fault_severity": 2,
        "resource_type": "resource_type 2",
    }

    final_state = run_pipeline(
        sample_ml_output,
        sample_fault,
        # first two hypotheses fail, third one fixes it:
        feedback_queue=[False, False, True],
    )

    print()
    print("=" * 70)
    print("FINAL PIPELINE STATE".center(70))
    print("=" * 70)
    print("Status        :", final_state.get("status"))
    print("Attempts used :", final_state.get("attempt", 0) + 1)
    print("Ticket        :", final_state.get("ticket"))
    print("Escalation    :", final_state.get("escalation"))
    print("Memory saved  :", final_state.get("memory_saved", False))
