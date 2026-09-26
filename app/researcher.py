"""
Researcher node.

Executes pending steps in state.plan (ACT), records what came back
(OBSERVE), and folds results into findings/sources/tool_history (UPDATE
STATE).

Phase 3 change: a step is no longer always run through web_search. Each
step is routed, via `app.tool_selector.choose_tool`, to the tool that
fits its shape:
  - a plain research question           -> web_search (the default)
  - a step that names a specific URL    -> page_reader (go deeper on one
                                            source than a search snippet)
  - a step that is/asks for a
    calculation                         -> calculator

This is the "Tool Selection -> Tool Execution -> Observation" part of the
target workflow. Phase 2's "the plan can grow between passes, so this
node may run more than once" behaviour is unchanged: it only processes
steps not already in `completed_steps`.
"""

from __future__ import annotations

from app.state import AgentState, Source, ToolCallRecord
from app.tool_selector import choose_tool, WEB_SEARCH, PAGE_READER, CALCULATOR
from app.tools.web_search import web_search
from app.tools.page_reader import read_page
from app.tools.calculator import calculate, CalculatorError

# How much of a fetched page's text to keep in `findings`. Full text can be
# several thousand characters; we only need enough for the reporter LLM to
# work with, not the entire article.
PAGE_TEXT_PREVIEW_CHARS = 1500


def _run_web_search(step: str, findings: list, sources: list, tool_history: list) -> None:
    print("[TOOL] web_search")
    print(f"[RESEARCH] Searching: {step}")
    try:
        results = web_search(step, max_results=3)
    except RuntimeError as e:
        print(f"[OBSERVE] web_search failed: {e}")
        tool_history.append(ToolCallRecord(tool="web_search", input=step, success=False, summary=str(e)))
        findings.append(f"Step '{step}': search failed ({e}).")
        return

    if not results:
        print("[OBSERVE] 0 results found")
        tool_history.append(ToolCallRecord(tool="web_search", input=step, success=True, summary="0 results"))
        findings.append(f"Step '{step}': no results found.")
        return

    print(f"[OBSERVE] {len(results)} result(s) found")
    tool_history.append(
        ToolCallRecord(tool="web_search", input=step, success=True, summary=f"{len(results)} result(s)")
    )

    step_findings = []
    for r in results:
        sources.append(Source(url=r.url, title=r.title, snippet=r.snippet))
        if r.snippet:
            step_findings.append(f"{r.title}: {r.snippet}")
    findings.append(f"Step '{step}':\n" + "\n".join(step_findings))


def _run_page_reader(step: str, url: str, findings: list, sources: list, tool_history: list) -> None:
    print("[TOOL] page_reader")
    print(f"[RESEARCH] Reading page: {url}")
    try:
        page = read_page(url)
    except RuntimeError as e:
        print(f"[OBSERVE] page_reader failed: {e}")
        tool_history.append(ToolCallRecord(tool="page_reader", input=url, success=False, summary=str(e)))
        findings.append(f"Step '{step}': could not read page ({e}).")
        return

    preview = page.text[:PAGE_TEXT_PREVIEW_CHARS]
    print(f"[OBSERVE] read {len(page.text)} char(s) from '{page.title or url}'")
    tool_history.append(
        ToolCallRecord(tool="page_reader", input=url, success=True, summary=f"{len(page.text)} char(s)")
    )
    sources.append(Source(url=page.url, title=page.title, snippet=preview[:200]))
    findings.append(f"Step '{step}':\n{page.title}\n{preview}")


def _run_calculator(step: str, expression: str, findings: list, tool_history: list) -> None:
    print("[TOOL] calculator")
    print(f"[RESEARCH] Calculating: {expression}")
    try:
        result = calculate(expression)
    except CalculatorError as e:
        print(f"[OBSERVE] calculator failed: {e}")
        tool_history.append(ToolCallRecord(tool="calculator", input=expression, success=False, summary=str(e)))
        findings.append(f"Step '{step}': calculation failed ({e}).")
        return

    print(f"[OBSERVE] result = {result}")
    tool_history.append(ToolCallRecord(tool="calculator", input=expression, success=True, summary=str(result)))
    findings.append(f"Step '{step}': result = {result}")


def research(state: AgentState) -> dict:
    findings = list(state.findings)
    sources = list(state.sources)
    tool_history = list(state.tool_history)
    completed_steps = list(state.completed_steps)

    pending_steps = [s for s in state.plan if s not in completed_steps]

    for step in pending_steps:
        tool_name, tool_input = choose_tool(step)

        if tool_name == CALCULATOR:
            _run_calculator(step, tool_input, findings, tool_history)
        elif tool_name == PAGE_READER:
            _run_page_reader(step, tool_input, findings, sources, tool_history)
        else:
            _run_web_search(step, findings, sources, tool_history)

        completed_steps.append(step)

    return {
        "findings": findings,
        "sources": sources,
        "tool_history": tool_history,
        "completed_steps": completed_steps,
        "current_step": len(completed_steps),
    }
