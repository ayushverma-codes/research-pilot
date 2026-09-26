"""
Reporter node.

Takes accumulated findings/sources and asks the LLM to synthesize a final,
coherent answer to the original user_goal. This is the "FINAL" stage.
Full multi-section report formatting is added in Phase 5 - MVP produces a
plain synthesized answer with a source list appended.
"""

from __future__ import annotations

from app.state import AgentState
from app.llm_provider import get_llm_client

REPORTER_SYSTEM_PROMPT = """You are a research report writer.
Using ONLY the findings provided, write a clear, coherent answer to the
user's research goal. Do not invent facts that are not supported by the
findings. If the findings are insufficient to fully answer the goal, say so
explicitly rather than guessing.
"""


def generate_report(state: AgentState) -> dict:
    llm = get_llm_client()

    findings_text = "\n\n".join(state.findings) if state.findings else "(no findings gathered)"
    prompt = (
        f"Research goal: {state.user_goal}\n\n"
        f"Findings gathered:\n{findings_text}\n\n"
        "Write the final answer now."
    )

    answer = llm.complete(prompt, system=REPORTER_SYSTEM_PROMPT, max_tokens=1200)

    sources_text = "\n".join(f"- {s.title} ({s.url})" for s in state.sources) or "(no sources)"
    report = f"{answer}\n\n---\nSources:\n{sources_text}"

    return {"final_report": report}
