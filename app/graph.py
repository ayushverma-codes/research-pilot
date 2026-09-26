"""
Graph wiring.

MVP is intentionally linear:

    planner -> researcher -> reporter

Each node is a plain function taking AgentState and returning a partial
update dict, per LangGraph's Pydantic-state convention. Phase 2 will replace
the straight researcher -> reporter edge with a conditional loop.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.planner import plan
from app.researcher import research
from app.reporter import generate_report


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan)
    builder.add_node("researcher", research)
    builder.add_node("reporter", generate_report)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "reporter")
    builder.add_edge("reporter", END)

    return builder.compile()
