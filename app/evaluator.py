"""
Critic / evidence-check node (Phase 2 loop, extended in Phase 4).

Phase 2 gave this node one job: given what's been found so far, is there
enough evidence to answer the user's goal, and if not, what should be
searched next? Phase 4 extends the same node into a fuller critic, asking
the LLM to also check *quality*, not just coverage:

  - Is the requested information covered? (sufficient / missing_information
    - unchanged from Phase 2)
  - Are important claims actually backed by a source, or just asserted?
  - Are the gathered sources relevant to the goal, or off-topic/noise?
  - Is the research as a whole still on-topic?

These extra checks come back as "issues" (a list of specific problems)
and "recommended_action" (a one-line statement of what to do about them),
and are stored in `state.critique` each pass purely for
visibility/logging/demo purposes - only `sufficient`/`missing_information`
(via `additional_queries`) actually drive the routing decision, exactly as
in Phase 2. Function/module names are kept as-is (`evaluate_evidence`,
`app.evaluator`) rather than renamed to `critic.py`, so existing Phase
2/3 tests keep working unchanged.

This is the "Check evidence" / "Critic" + "Missing info?" decision point in:

    Research -> Observe -> Critic / Evidence Check -> Enough evidence?
                                                ├─ No  -> Research again
                                                └─ Yes -> continue
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

EVALUATOR_SYSTEM_PROMPT = """You are a research evidence checker (critic).
Given a research goal and the findings gathered so far, judge the evidence
on two dimensions:

1. COVERAGE - is there enough evidence to write a good final answer?
2. QUALITY - independent of coverage, are there problems with what has
   been gathered:
   - unsupported claims: something findings assert with no source behind it
   - irrelevant/off-topic sources: a source that doesn't actually bear on
     the goal
   - the research drifting off-topic from the original goal

Respond with ONLY a JSON object, nothing else, in this exact shape:
{"sufficient": true or false, "missing_information": ["...", ...], "issues": ["...", ...], "recommended_action": "...", "additional_queries": ["...", ...]}

Rules:
- "missing_information": short descriptions of what's still missing for
  COVERAGE. Empty list if sufficient.
- "issues": short descriptions of QUALITY problems (unsupported claims,
  irrelevant sources, off-topic drift). Empty list if none found. A
  finding can be sufficient for coverage and still have issues.
- "recommended_action": one short sentence: either the single next action
  to take (e.g. "Search official pricing page"), or, if nothing more is
  needed, state that the evidence is sufficient and ready for the report.
- "additional_queries": 1 to 3 concrete next actions that would fill the
  coverage gaps or resolve the issues above. Empty list if sufficient.
  Usually this is a new web search query, but it can instead be:
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
  even if not every minor detail is present. Quality "issues" alone
  (without missing_information) do not force sufficient=false unless they
  are serious enough that the report would make an unsupported claim.
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
        critique = {
            "sufficient": True,
            "missing_information": [],
            "issues": [],
            "recommended_action": f"Max iterations ({MAX_ITERATIONS}) reached; proceeding to report as-is.",
        }
        return {"iteration": iteration, "missing_information": [], "critique": critique}

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

    # A critic call occasionally comes back empty or without valid JSON
    # (observed live: an empty string from the LLM). That's usually a
    # one-off blip, not a real "the model has an opinion" response, so
    # retry once with a sharper reminder before treating it as a genuine
    # evaluator failure. This intentionally mirrors GroqClient's own
    # bounded-retry pattern in llm_provider.py rather than inventing a new
    # retry style.
    MAX_ATTEMPTS = 2
    result = None
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\n\n(Your previous reply was empty or was not valid JSON. "
                "Respond with ONLY the JSON object, nothing else.)"
            )
        try:
            raw = llm.complete(attempt_prompt, system=EVALUATOR_SYSTEM_PROMPT, max_tokens=700)
            result = _extract_json_object(raw)
            break
        except Exception as e:  # noqa: BLE001 — a broken evaluator must not kill the run
            last_error = e
            if attempt < MAX_ATTEMPTS:
                print(f"[CRITIC] Evaluation attempt {attempt} failed ({e}) — retrying once.")

    if result is None:
        print(f"[CRITIC] Evaluation failed after {MAX_ATTEMPTS} attempt(s) ({last_error}) — assuming evidence is sufficient.")
        critique = {
            "sufficient": True,
            "missing_information": [],
            "issues": [f"Critic evaluation itself failed after {MAX_ATTEMPTS} attempt(s): {last_error}"],
            "recommended_action": "Evaluation failed; proceeding to report on existing findings.",
        }
        return {"iteration": iteration, "missing_information": [], "critique": critique}

    sufficient = bool(result.get("sufficient", True))
    missing = [str(m).strip() for m in result.get("missing_information", []) if str(m).strip()]
    issues = [str(i).strip() for i in result.get("issues", []) if str(i).strip()]
    recommended_action = str(result.get("recommended_action", "")).strip()
    new_queries = [str(q).strip() for q in result.get("additional_queries", []) if str(q).strip()]

    # Drop any proposed URL the LLM invented rather than copied from
    # state.sources - see _drop_hallucinated_urls' docstring.
    known_urls = {s.url for s in state.sources}
    new_queries = _drop_hallucinated_urls(new_queries, known_urls)

    # Never re-queue a query that's already planned or already run.
    already_known = set(state.plan) | set(state.completed_steps)
    new_queries = [q for q in new_queries if q not in already_known]

    if issues:
        print(f"[CRITIC] Quality issues noted: {issues}")

    critique = {
        "sufficient": sufficient or not new_queries,
        "missing_information": missing,
        "issues": issues,
        "recommended_action": recommended_action,
    }

    if sufficient or not new_queries:
        print("[CRITIC] Evidence sufficient — continuing to report.")
        return {"iteration": iteration, "missing_information": [], "critique": critique}

    print(f"[CRITIC] Evidence incomplete: {missing}")
    print(f"[DECISION] Additional research required — queueing {len(new_queries)} new search(es): {new_queries}")

    return {
        "iteration": iteration,
        "missing_information": missing,
        "plan": state.plan + new_queries,
        "critique": critique,
    }


def needs_more_research(state: AgentState) -> str:
    """Routing function for the conditional edge out of the evaluator node."""
    if state.missing_information and state.iteration <= MAX_ITERATIONS:
        return "researcher"
    return "reporter"
