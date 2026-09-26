"""
Graph wiring.

Phase 1 was intentionally linear (planner -> researcher -> reporter).
Phase 2 replaced the fixed researcher -> reporter edge with a real
conditional loop, and Phase 4 extends the same node into a fuller
critic (coverage + quality checks, see app/evaluator.py):

    planner -> researcher -> evaluator/critic -+-> researcher   (evidence insufficient)
                                                '-> reporter      (evidence sufficient,
                                                                    or max iterations hit)

The LLM-driven `evaluator` (critic) node decides which branch to take each
time, based on current state (findings so far, iteration count) — the
agent does not blindly execute a fixed sequence. The node is still named
`evaluator` / registered as "evaluator" here for continuity with Phase 2/3
tests; conceptually it is the "Critic / Evidence Check" box in the target
workflow.
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
