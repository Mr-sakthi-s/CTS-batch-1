import unittest
from types import SimpleNamespace

from ai_engine.agents import rca_engine


class TestRcaPipeline(unittest.TestCase):
    def test_generate_rca_agentic_builds_three_candidates_when_historical_context_is_empty(self):
        ml_output = {
            "severity_type": "Major",
            "resource_type": "Router",
            "event_types": ["Packet Loss", "Latency"],
            "log_features": ["Queue Depth", "CPU"],
            "predicted_fault_severity": "Critical",
            "volume": 500,
        }

        knowledge = """
        ROOT CAUSE: Network Congestion
        DESCRIPTION: Demand exceeds capacity.
        REMEDIATION: Check utilization and increase bandwidth.

        ROOT CAUSE: Backhaul Congestion
        DESCRIPTION: Link overload.
        REMEDIATION: Inspect backhaul utilization and redistribute traffic.

        ROOT CAUSE: Router CPU Saturation
        DESCRIPTION: Router resource exhaustion.
        REMEDIATION: Analyze CPU utilization and reduce traffic load.
        """

        original_knowledge_retriever = rca_engine.knowledge_retriever
        original_pattern_retriever = rca_engine.pattern_retriever
        original_ollama_check = rca_engine.check_ollama_connection
        original_ollama_call = rca_engine.call_ollama

        try:
            rca_engine.knowledge_retriever = SimpleNamespace(
                invoke=lambda *_args, **_kwargs: [SimpleNamespace(page_content=knowledge)]
            )
            rca_engine.pattern_retriever = SimpleNamespace(invoke=lambda *_args, **_kwargs: [])
            rca_engine.check_ollama_connection = lambda: True

            def fake_call(prompt):
                if "PATTERN ANALYST" in prompt:
                    return ""
                return (
                    "Resolved root cause: Network Congestion\n"
                    "Resolution: Check utilization and increase bandwidth\n"
                    "Confidence: 95%\n"
                    "Evidence: Knowledge match for network congestion"
                )

            rca_engine.call_ollama = fake_call

            result = rca_engine.generate_rca_agentic(ml_output)

            self.assertTrue(result["ranked_causes"])
            self.assertEqual(len(result["ranked_causes"]), 3)
            self.assertEqual(
                {candidate["root_cause"] for candidate in result["ranked_causes"]},
                {"Network Congestion", "Backhaul Congestion", "Router CPU Saturation"},
            )
        finally:
            rca_engine.knowledge_retriever = original_knowledge_retriever
            rca_engine.pattern_retriever = original_pattern_retriever
            rca_engine.check_ollama_connection = original_ollama_check
            rca_engine.call_ollama = original_ollama_call


if __name__ == "__main__":
    unittest.main()
