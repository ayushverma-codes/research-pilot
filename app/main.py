"""
CLI entrypoint for ResearchPilot MVP.

Usage:
    python -m app.main "your research question"
    python -m app.main            # will prompt interactively
"""

from __future__ import annotations

import sys

from app.graph import build_graph
from app.state import AgentState
from app.tools.report_writer import write_report


def run(user_goal: str) -> AgentState:
    graph = build_graph()
    initial_state = AgentState(user_goal=user_goal)
    result = graph.invoke(initial_state)
    return AgentState(**result)


def main() -> None:
    if len(sys.argv) > 1:
        user_goal = " ".join(sys.argv[1:])
    else:
        user_goal = input("Enter your research question: ").strip()

    if not user_goal:
        print("Error: research question cannot be empty.")
        sys.exit(1)

    print(f"[GOAL] {user_goal}")

    try:
        state = run(user_goal)
    except Exception as e:  # noqa: BLE001 - top-level CLI error boundary
        print(f"[ERROR] Agent run failed: {e}")
        sys.exit(1)

    print(f"\n[SUMMARY] {len(state.plan)} total step(s) planned, "
          f"{len(state.completed_steps)} completed, "
          f"{len(state.sources)} source(s) gathered, "
          f"{state.iteration} research iteration(s)")
    print("\n=== FINAL REPORT ===\n")
    print(state.final_report)

    print("[TOOL] report_writer")
    path = write_report(state.user_goal, state.final_report)
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
