"""
CLI entrypoint for ResearchPilot MVP.

Usage:
    python -m app.main "your research question"
    python -m app.main            # will prompt interactively
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from app.graph import build_graph
from app.state import AgentState

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run(user_goal: str) -> AgentState:
    graph = build_graph()
    initial_state = AgentState(user_goal=user_goal)
    result = graph.invoke(initial_state)
    return AgentState(**result)


def save_report(state: AgentState) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"report_{timestamp}.md"
    path.write_text(
        f"# Research Report\n\n**Goal:** {state.user_goal}\n\n{state.final_report}\n",
        encoding="utf-8",
    )
    return path


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

    print(f"[PLAN] {len(state.plan)} step(s): {state.plan}")
    print(f"[RESEARCH] {len(state.completed_steps)} step(s) completed, "
          f"{len(state.sources)} source(s) gathered")
    print("\n=== FINAL REPORT ===\n")
    print(state.final_report)

    path = save_report(state)
    print(f"\n[SAVED] {path}")


if __name__ == "__main__":
    main()
