# Telecom Fault-Resolution Agent (LangGraph)

All five standalone agent files (`rca_engine.py`, `dispatch_agent.py`,
`feedback_agent.py`, `escalation_agent.py`, `memory_agent.py`) are now
consolidated into a single LangGraph package:

```
agent/
├── __init__.py     # public API: build_graph, run_pipeline, AgentState
├── state.py        # AgentState / Ticket / RCACandidate TypedDicts
├── prompts.py      # Pattern-Analyst + Final-RCA prompt templates
├── nodes.py        # all agent logic as functions + LangGraph node wrappers
└── graph.py        # StateGraph wiring, routers, compiled graph, demo
```

## Graph flow

```
START → rca → dispatch ─┬→ verify → feedback ─┬→ memory → END      (fixed)
                        │     ↑               ├→ verify            (retry next hypothesis)
                        │     └───────────────┘
                        └→ escalation → END                        (no technician /
                                                                    all 3 hypotheses failed)
```

- **rca** — agentic RCA (semantic mapping → knowledge RAG → pattern RAG →
  pattern-analyst LLM → final RCA LLM → parser). Always yields exactly 3
  ranked hypotheses and opens a ticket.
- **dispatch** — nearest-region technician + spare-part assignment for the
  current hypothesis. If no technician exists anywhere → escalation.
- **verify** — did the attempted fix work? Simulated via `feedback_queue`,
  or human-in-the-loop (see below).
- **feedback** — success → CLOSED; failure → next ranked hypothesis
  (max 3); exhausted → ESCALATE.
- **memory** — self-learning: writes the successful resolution to
  `resolution_history.csv` and the `telecom_patterns` vector DB.
- **escalation** — hands the ticket to `NOC_ENGINEERING_TEAM`.

## Usage

```python
from agent import run_pipeline

result = run_pipeline(
    ml_output={
        "severity_type": "severity_type 4",
        "resource_type": "resource_type 2",
        "event_types": ["event_type 24", "event_type 22"],
        "log_features": ["feature 265", "feature 271"],
        "predicted_fault_severity": 2,
        "volume": 14,
    },
    fault={
        "id": 118,
        "location": "location 118",
        "fault_severity": 2,
        "resource_type": "resource_type 2",
    },
    feedback_queue=[False, False, True],  # simulate: 3rd fix works
)

print(result["status"])        # CLOSED
print(result["memory_saved"])  # True
```

Or run the built-in demo:

```bash
python -m agent.graph
```

## Human-in-the-loop verification (production)

```python
from langgraph.checkpoint.memory import MemorySaver
from agent import build_graph

graph = build_graph(checkpointer=MemorySaver(), human_in_loop=True)
config = {"configurable": {"thread_id": "ticket-118"}}

graph.invoke({"ml_output": ..., "fault": ...}, config)   # pauses at verify
graph.invoke({"fixed": True}, config)                    # resume with outcome
```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_DATA_DIR` | `data` | folder for reference CSVs |
| `TECHNICIANS_CSV` | `data/technicians.csv` | technician roster |
| `SPARE_PARTS_CSV` | `data/spare_parts.csv` | spare-part inventory |
| `RESOLUTION_FILE` | `resolution_history.csv` | self-learning CSV |
| `VECTOR_DB_PATH` | `vector_db` | Chroma persist directory |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF id or local snapshot path |
| `OLLAMA_MODEL` | `telecom-copilot` | Ollama model name |

The hardcoded Windows paths from the original files were replaced with
these env-driven defaults, so the package runs on any machine.

## Requirements

```
langgraph
pandas
langchain-chroma
langchain-huggingface
langchain-ollama
python-dotenv        # optional
psycopg2-binary      # optional (Postgres dispatch records)
```

Heavy dependencies are lazily imported, so the graph can be built and
unit-tested without Chroma/Ollama installed.
