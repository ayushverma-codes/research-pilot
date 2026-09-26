"""
Evaluator node (Phase 2 — lightweight evidence check).

This is deliberately NOT the full Critic from Phase 4 (structured
sufficient/issues/recommended_action schema with source-relevance checks).
For Phase 2 it only has to answer one question well: given what's been
found so far, is there enough evidence to answer the user's goal, and if
not, what should be searched next? Phase 4 will replace/extend this with a
fuller critic node.

This is the "Check evidence" + "Missing information?" decision point in:

    Research -> Observe -> Check evidence -> Missing info?
                                                ├─ Yes -> Research again
                                                └─ No  -> continue
"""

from __future__ import annotations

import json
import os
import re

from app.state import AgentState
from app.llm_provider import get_llm_client

# Hard ceiling on research loops, so a stubborn/ambiguous goal can never spin
# forever. Configurable for experimentation, but always enforced.
MAX_ITERATIONS = int(os.getenv("RESEARCHPILOT_MAX_ITERATIONS", "3"))

EVALUATOR_SYSTEM_PROMPT = """You are a research evidence checker.
Given a research goal and the findings gathered so far, decide whether
there is enough evidence to write a good final answer.

Respond with ONLY a JSON object, nothing else, in this exact shape:
{"sufficient": true or false, "missing_information": ["...", ...], "additional_queries": ["...", ...]}

Rules:
- "missing_information": short descriptions of what's still missing. Empty
  list if sufficient.
- "additional_queries": 1 to 3 concrete next actions that would fill the
  gaps. Empty list if sufficient. Usually this is a new web search query,
  but it can instead be:
    - one of the exact URLs listed under "Known sources" below, copied
      character-for-character, if that source needs to be read in more
      depth than its search snippet gives you. NEVER construct, guess,
      or modify a URL (e.g. from a document title you saw in the
      findings) - only ever copy one that is already listed verbatim
      under "Known sources"; or
    - "Calculate: <expression>" with a plain arithmetic expression built
      from numbers already present in the findings, if a computation
      (e.g. a price difference) would help answer the goal.
- Never propose an action that has already been run.
- If the findings already reasonably cover the goal, set sufficient=true
  even if not every minor detail is present.
"""

URL_RE = re.compile(r"https?://\S+")


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in evaluator output: {text!r}")
    return json.loads(match.group(0))


def _format_known_sources(state: AgentState) -> str:
    """
    Render state.sources as a "- title: url" list for the prompt, deduped
    by URL, so the evaluator has an exact set of real URLs to copy from
    instead of reconstructing one from a title mentioned in the findings.
    """
    lines = []
    seen = set()
    for s in state.sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        lines.append(f"- {s.title}: {s.url}")
    return "\n".join(lines) if lines else "(no sources yet)"


def _drop_hallucinated_urls(queries: list, known_urls: set) -> list:
    """
    Defense-in-depth on top of the prompt instruction: drop any proposed
    action that contains a URL not present in `known_urls`. The LLM is
    told to only ever copy a URL verbatim from "Known sources", but
    prompt instructions aren't a guarantee - this stops a fabricated or
    reconstructed URL (which will just 404 in page_reader) from ever
    reaching the researcher. Non-URL actions (search queries,
    "Calculate: ...") pass through unchanged.
    """
    kept = []
    for q in queries:
        match = URL_RE.search(q)
        if match:
            url = match.group(0).rstrip(").,;\"'")
            if url not in known_urls:
                print(f"[CRITIC] Dropping proposed URL not in known sources (likely fabricated): {q}")
                continue
        kept.append(q)
    return kept


def evaluate_evidence(state: AgentState) -> dict:
    """
    Decide whether current findings are enough to report, or whether another
    research pass is needed. Returns a state update dict.

    Contract with the router (`needs_more_research`, below): this function
    only ever returns a non-empty `missing_information` when it is also
    handing back new queries in `plan` for the researcher to act on. So the
    router can make its decision by looking at `missing_information` alone.
    """
    iteration = state.iteration + 1

    if iteration > MAX_ITERATIONS:
        print(f"[DECISION] Max iterations ({MAX_ITERATIONS}) reached — proceeding to report.")
        return {"iteration": iteration, "missing_information": []}

    llm = get_llm_client()
    findings_text = "\n\n".join(state.findings) if state.findings else "(no findings yet)"
    already_searched = ", ".join(state.completed_steps) or "(none)"
    known_sources_text = _format_known_sources(state)
    prompt = (
        f"Research goal: {state.user_goal}\n\n"
        f"Already searched: {already_searched}\n\n"
        f"Findings gathered:\n{findings_text}\n\n"
        f"Known sources (title: exact URL - copy one of these verbatim if "
        f"you want a page read in depth; never construct your own):\n"
        f"{known_sources_text}\n\n"
        "Is this enough evidence? Respond with the JSON object now."
    )

    try:
        raw = llm.complete(prompt, system=EVALUATOR_SYSTEM_PROMPT, max_tokens=700)
        result = _extract_json_object(raw)
    except Exception as e:  # noqa: BLE001 — a broken evaluator must not kill the run
        print(f"[CRITIC] Evaluation failed ({e}) — assuming evidence is sufficient.")
        return {"iteration": iteration, "missing_information": []}

    sufficient = bool(result.get("sufficient", True))
    missing = [str(m).strip() for m in result.get("missing_information", []) if str(m).strip()]
    new_queries = [str(q).strip() for q in result.get("additional_queries", []) if str(q).strip()]

    # Drop any proposed URL the LLM invented rather than copied from
    # state.sources - see _drop_hallucinated_urls' docstring.
    known_urls = {s.url for s in state.sources}
    new_queries = _drop_hallucinated_urls(new_queries, known_urls)

    # Never re-queue a query that's already planned or already run.
    already_known = set(state.plan) | set(state.completed_steps)
    new_queries = [q for q in new_queries if q not in already_known]

    if sufficient or not new_queries:
        print("[CRITIC] Evidence sufficient — continuing to report.")
        return {"iteration": iteration, "missing_information": []}

    print(f"[CRITIC] Evidence incomplete: {missing}")
    print(f"[DECISION] Additional research required — queueing {len(new_queries)} new search(es): {new_queries}")

    return {
        "iteration": iteration,
        "missing_information": missing,
        "plan": state.plan + new_queries,
    }


def needs_more_research(state: AgentState) -> str:
    """Routing function for the conditional edge out of the evaluator node."""
    if state.missing_information and state.iteration <= MAX_ITERATIONS:
        return "researcher"
    return "reporter"
