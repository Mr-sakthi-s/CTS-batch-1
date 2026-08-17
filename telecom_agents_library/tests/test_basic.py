from telecom_agents.escalation_agent import escalate
from telecom_agents.feedback_agent import process_feedback
from telecom_agents.dispatch_agent import derive_region

def test_derive_region():
    assert derive_region("location 118") == "region_8"

def test_escalate():
    assert escalate({"ticket_id": 123})["status"] == "ESCALATED"

def test_feedback_success():
    ticket = {"status": "OPEN", "attempt": 0, "ranked_causes": [{}, {}, {}]}
    assert process_feedback(ticket, True)["status"] == "CLOSED"
