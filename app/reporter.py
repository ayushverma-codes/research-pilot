"""
Reporter node.

Phase 1-4 produced a plain synthesized answer with a flat source list
appended. Phase 5 upgrades this into a structured, multi-section report:

    # Research Report
    ## Executive Summary      <- LLM (facts, direct answer)
    ## Research Question      <- deterministic (state.user_goal)
    ## Methodology            <- deterministic (built from tool_history/
                                  iteration/sources - what actually ran)
    ## Key Findings           <- LLM (facts only, drawn from findings)
    ## Comparison / Analysis  <- LLM (interpretation/synthesis, kept
                                  separate from Key Findings so facts and
                                  analysis are never mixed together)
    ## Limitations            <- deterministic base (missing_information /
                                  critic issues actually recorded during
                                  the run) + LLM additions
    ## Sources                <- deterministic, de-duplicated

Only the narrative sections (Executive Summary, Key Findings,
Comparison / Analysis, Limitations-additions) come from the LLM, in a
single call. Everything else is assembled directly from `state`, so the
report can never claim research happened that the run didn't actually do.
"""

from __future__ import annotations

import re

from app.state import AgentState
from app.llm_provider import get_llm_client

REPORTER_SYSTEM_PROMPT = """You are a research report writer.
Using ONLY the findings provided, write the analytical portion of a
research report as EXACTLY these four markdown sections, in this order,
with these exact headings:

## Executive Summary
2-4 sentences giving the direct answer to the research question, if the
findings support one.

## Key Findings
Bullet points. Each bullet must be a specific fact or data point that is
actually present in the findings below. Do not add interpretation here.

## Comparison / Analysis
A short synthesis, comparison, or interpretation of the findings. If no
real comparison applies, briefly explain what the findings indicate
overall instead. Keep opinion/interpretation here, not in Key Findings.

## Limitations
Bullet points describing what remains unknown, unsupported, or missing
from the findings below.

Rules:
- Do not invent facts that are not supported by the findings.
- If the findings are empty or clearly insufficient to answer the
  research question, say so explicitly in the Executive Summary, and
  list it as a limitation, rather than guessing.
- Do not claim that browsing, searching, or reading happened beyond what
  the findings below actually show.
- Output ONLY the four sections above (headings + content) - nothing
  before the first heading and nothing after the last section.
"""

_HEADING_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")


def _split_sections(text: str) -> dict:
    """
    Split LLM markdown output on '## Heading' lines into
    {heading: body}. Best-effort: if the LLM didn't follow the requested
    format, this returns an empty dict and the caller falls back to
    treating the whole response as the executive summary.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections: dict = {}
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    return sections


_MAX_BULLET_CHARS = 220


def _clean_bullet_text(text: str) -> str:
    """
    Collapse a piece of arbitrary text (which may contain newlines, or be
    a raw exception message rather than a short human-readable note - see
    evaluator.py's "Critic evaluation itself failed" issue) into a single
    readable line, and cap its length. Defense-in-depth: state.critique
    is meant to hold short strings, but nothing enforces that upstream,
    and this text goes straight into the user-facing report, so it's
    sanitized again here rather than trusting the source.
    """
    collapsed = " ".join(str(text).split())
    if len(collapsed) > _MAX_BULLET_CHARS:
        return collapsed[:_MAX_BULLET_CHARS] + "...(truncated)"
    return collapsed


def _build_methodology(state: AgentState) -> str:
    """Deterministic account of what the agent actually did this run."""
    tool_counts: dict = {}
    for rec in state.tool_history:
        tool_counts[rec.tool] = tool_counts.get(rec.tool, 0) + 1

    if tool_counts:
        tools_text = ", ".join(f"{name} x{count}" for name, count in tool_counts.items())
    else:
        tools_text = "no tools were used"

    lines = [
        f"This report was produced by an autonomous research agent over "
        f"{state.iteration} research iteration(s), executing "
        f"{len(state.tool_history)} tool call(s) ({tools_text}) across "
        f"{len(state.completed_steps)} research step(s), gathering "
        f"{len(state.sources)} source reference(s)."
    ]
    if state.critique and state.critique.get("recommended_action"):
        lines.append(f"Final evidence check: {_clean_bullet_text(state.critique['recommended_action'])}")
    return "\n\n".join(lines)


def _build_limitations(state: AgentState, llm_limitations: str) -> str:
    """
    Deterministic limitations (actually-recorded gaps/issues from the run)
    combined with the LLM's own limitations bullets. Deterministic bullets
    come first since they reflect the run's real state, not a guess.
    """
    bullets = []
    for m in state.missing_information:
        bullets.append(f"- Not covered: {_clean_bullet_text(m)}")
    if state.critique:
        for issue in state.critique.get("issues", []):
            bullets.append(f"- Quality issue noted during research: {_clean_bullet_text(issue)}")
    if not state.sources:
        bullets.append("- No sources were successfully gathered; findings below may be incomplete or absent.")

    base = "\n".join(bullets)
    combined = "\n\n".join(part for part in [base, llm_limitations] if part)
    return combined or "- No significant limitations identified."


def _build_sources(state: AgentState) -> str:
    """De-duplicated, ordered source list (by URL)."""
    seen = set()
    lines = []
    for s in state.sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        title = s.title or s.url
        lines.append(f"- {title} ({s.url})")
    return "\n".join(lines) if lines else "(no sources)"


def generate_report(state: AgentState) -> dict:
    llm = get_llm_client()

    findings_text = "\n\n".join(state.findings) if state.findings else "(no findings gathered)"
    prompt = (
        f"Research question: {state.user_goal}\n\n"
        f"Findings gathered:\n{findings_text}\n\n"
        "Write the four sections now."
    )

    raw = llm.complete(prompt, system=REPORTER_SYSTEM_PROMPT, max_tokens=2000)
    sections = _split_sections(raw)

    if sections:
        exec_summary = sections.get("Executive Summary", "").strip()
        key_findings = sections.get("Key Findings", "").strip()
        comparison = sections.get("Comparison / Analysis", "").strip()
        llm_limitations = sections.get("Limitations", "").strip()
    else:
        # LLM didn't follow the requested format - fall back to putting
        # its whole response in the summary rather than losing content.
        exec_summary = raw.strip()
        key_findings = ""
        comparison = ""
        llm_limitations = ""

    methodology = _build_methodology(state)
    limitations = _build_limitations(state, llm_limitations)
    sources_text = _build_sources(state)

    report = (
        "# Research Report\n\n"
        f"## Executive Summary\n\n{exec_summary or '(no summary generated)'}\n\n"
        f"## Research Question\n\n{state.user_goal}\n\n"
        f"## Methodology\n\n{methodology}\n\n"
        f"## Key Findings\n\n{key_findings or '(no findings gathered)'}\n\n"
        f"## Comparison / Analysis\n\n{comparison or '(not enough findings for a comparison/analysis)'}\n\n"
        f"## Limitations\n\n{limitations}\n\n"
        f"## Sources\n\n{sources_text}\n"
    )

    return {"final_report": report}
