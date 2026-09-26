"""
Researcher node.

Executes every step in state.plan using the web_search tool (ACT), records
what was returned (OBSERVE), and folds results into findings/sources/
tool_history (UPDATE STATE). MVP is a single linear pass over the plan -
no re-planning yet (that's Phase 2).
"""

from __future__ import annotations

from app.state import AgentState, Source, ToolCallRecord
from app.tools.web_search import web_search


def research(state: AgentState) -> dict:
    findings = list(state.findings)
    sources = list(state.sources)
    tool_history = list(state.tool_history)
    completed_steps = list(state.completed_steps)

    for step in state.plan:
        try:
            results = web_search(step, max_results=3)
        except RuntimeError as e:
            tool_history.append(
                ToolCallRecord(tool="web_search", input=step, success=False, summary=str(e))
            )
            findings.append(f"Step '{step}': search failed ({e}).")
            completed_steps.append(step)
            continue

        if not results:
            tool_history.append(
                ToolCallRecord(tool="web_search", input=step, success=True, summary="0 results")
            )
            findings.append(f"Step '{step}': no results found.")
            completed_steps.append(step)
            continue

        tool_history.append(
            ToolCallRecord(
                tool="web_search",
                input=step,
                success=True,
                summary=f"{len(results)} result(s)",
            )
        )

        step_findings = []
        for r in results:
            sources.append(Source(url=r.url, title=r.title, snippet=r.snippet))
            if r.snippet:
                step_findings.append(f"{r.title}: {r.snippet}")

        findings.append(f"Step '{step}':\n" + "\n".join(step_findings))
        completed_steps.append(step)

    return {
        "findings": findings,
        "sources": sources,
        "tool_history": tool_history,
        "completed_steps": completed_steps,
        "current_step": len(completed_steps),
    }
