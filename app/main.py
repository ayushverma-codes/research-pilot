"""
CLI entrypoint for ResearchPilot MVP.

Usage:
    python -m app.main "your research question"
    python -m app.main                 # will prompt interactively
    python -m app.main --demo          # run a canned demo question with
                                        # extra banners for judges/viewers

Phase 10 note: the underlying agent run is identical in all three modes -
`--demo` only changes what's printed around it (banners, a preset
question) and never alters graph/agent behavior itself. Every stage's
own node already logs its own concise action/reason summary as it runs
(see app/planner.py, app/researcher.py, app/evaluator.py,
app/reporter.py); this file just frames that trace with a clear
start/end so someone watching the terminal can follow

    USER TASK -> PLAN -> TOOL CALL -> OBSERVATION -> STATE UPDATE ->
    CRITIC -> (RESEARCH AGAIN if needed) -> FINAL REPORT

without needing to read code. No hidden chain-of-thought is ever printed
here or in any node - only these short, human-readable action/reason
lines.
"""

from __future__ import annotations

import sys

from app.graph import build_graph
from app.state import AgentState
from app.tools.report_writer import write_report
from app.guardrails import GuardrailError, validate_task

# A single built-in question for `--demo` mode. Chosen because it needs
# real comparison across at least two named things (not a one-shot lookup),
# which tends to give the critic something concrete to ask for a second
# pass on - but the agent's actual behavior on it is never hardcoded or
# faked; it runs the same graph as any other question.
DEMO_QUESTION = (
    "Compare the pricing and context window size of the latest Claude "
    "and GPT models."
)

_RULE = "=" * 70


def _banner(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def run(user_goal: str) -> AgentState:
    graph = build_graph()
    initial_state = AgentState(user_goal=user_goal)
    result = graph.invoke(initial_state)
    return AgentState(**result)


def main() -> None:
    demo_mode = "--demo" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--demo"]

    if demo_mode:
        user_goal = DEMO_QUESTION
    elif args:
        user_goal = " ".join(args)
    else:
        user_goal = input("Enter your research question: ").strip()

    try:
        user_goal = validate_task(user_goal)
    except GuardrailError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if demo_mode:
        _banner("RESEARCHPILOT DEMO")
        print("Autonomous research agent: plan -> act -> observe -> update state")
        print("-> critic -> re-plan if needed -> final report.")
        print("Every line below is this run's real, unedited agent trace.")

    _banner(f"USER TASK\n{user_goal}")

    try:
        state = run(user_goal)
    except Exception as e:  # noqa: BLE001 - top-level CLI error boundary
        print(f"[ERROR] Agent run failed: {e}")
        sys.exit(1)

    if not state.final_report or not state.final_report.strip():
        # Defensive net only: every current code path through reporter.py
        # always emits a fully-headed report (with "(no findings
        # gathered)"-style placeholders), so this should be unreachable.
        # Guard it anyway rather than saving/printing a blank file.
        print("[ERROR] Agent run produced an empty report; nothing to save.")
        sys.exit(1)

    _banner("FINAL REPORT")
    print(state.final_report)

    print("[TOOL] report_writer")
    path = write_report(state.user_goal, state.final_report)
    print(f"[SAVED] {path}")

    _banner(
        f"SUMMARY: {len(state.plan)} step(s) planned, "
        f"{len(state.completed_steps)} completed, "
        f"{len(state.sources)} source(s) gathered, "
        f"{state.iteration} research iteration(s)"
    )


if __name__ == "__main__":
    main()

