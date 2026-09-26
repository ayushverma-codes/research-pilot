"""
Graph wiring.

Phase 1 was intentionally linear (planner -> researcher -> reporter).
Phase 2 replaces the fixed researcher -> reporter edge with a real
conditional loop:

    planner -> researcher -> evaluator -+-> researcher   (evidence insufficient)
                                         '-> reporter      (evidence sufficient,
                                                             or max iterations hit)

The LLM-driven `evaluator` node decides which branch to take each time,
based on current state (findings so far, iteration count) — the agent does
not blindly execute a fixed sequence.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.planner import plan
from app.researcher import research
from app.evaluator import evaluate_evidence, needs_more_research
from app.reporter import generate_report


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan)
    builder.add_node("researcher", research)
    builder.add_node("evaluator", evaluate_evidence)
    builder.add_node("reporter", generate_report)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        needs_more_research,
        {"researcher": "researcher", "reporter": "reporter"},
    )
    builder.add_edge("reporter", END)

    return builder.compile()
