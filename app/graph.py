"""
Graph wiring.

Phase 1 was intentionally linear (planner -> researcher -> reporter).
Phase 2 replaced the fixed researcher -> reporter edge with a real
conditional loop, and Phase 4 extends the same node into a fuller
critic (coverage + quality checks, see app/evaluator.py). Phase 6 adds a
`memory_writer` node after the report is generated, so every run's
experience (successful/failed queries, useful source domains) is saved
for the *next* run's planner to draw on (see app/memory.py):

    planner -> researcher -> evaluator/critic -+-> researcher   (evidence insufficient)
                                                '-> reporter -> memory_writer -> END
                                                     (evidence sufficient,
                                                      or max iterations hit)

The LLM-driven `evaluator` (critic) node decides which branch to take each
time, based on current state (findings so far, iteration count) — the
agent does not blindly execute a fixed sequence. The node is still named
`evaluator` / registered as "evaluator" here for continuity with Phase 2/3
tests; conceptually it is the "Critic / Evidence Check" box in the target
workflow. `memory_writer` is a pure side-effecting node (writes to disk,
returns no state update) rather than something the routing logic depends
on, so it can't affect the research loop itself.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.planner import plan
from app.researcher import research
from app.evaluator import evaluate_evidence, needs_more_research
from app.reporter import generate_report
from app.memory import record_run


def _write_memory(state: AgentState) -> dict:
    """Graph node wrapper around app.memory.record_run (Phase 6)."""
    record_run(state)
    print("[MEMORY] Saved this run's queries/sources for future planning.")
    return {}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan)
    builder.add_node("researcher", research)
    builder.add_node("evaluator", evaluate_evidence)
    builder.add_node("reporter", generate_report)
    builder.add_node("memory_writer", _write_memory)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        needs_more_research,
        {"researcher": "researcher", "reporter": "reporter"},
    )
    builder.add_edge("reporter", "memory_writer")
    builder.add_edge("memory_writer", END)

    return builder.compile()
