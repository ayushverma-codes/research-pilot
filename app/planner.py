"""
Planner node.

Takes the user's research goal and asks the LLM to produce a short list of
concrete research steps (as JSON). This is the "PLAN" stage of
PLAN -> ACT -> OBSERVE -> UPDATE STATE -> FINAL.

Phase 6 addition: before building the prompt, the planner retrieves any
related past runs from `app.memory` (previous experience -> better
planning, not model training - see that module's docstring) and folds a
short hint block into the prompt. This is advisory context only; the LLM
still produces its own plan and nothing here is treated as fact.
"""

from __future__ import annotations

import json
import re

from app.state import AgentState
from app.llm_provider import get_llm_client
from app.memory import get_planning_hint

PLANNER_SYSTEM_PROMPT = """You are a research planning assistant.
Given a research goal, break it down into 3 to 5 concrete, ordered research
steps that, together, would let someone answer the goal using web search.
Each step must be a short, self-contained action (e.g. a specific thing to
search for or look up).

If notes from related past research runs are provided, use them only as
hints: prefer query phrasings similar to ones that worked before, and
avoid repeating ones that failed or found nothing - but still produce a
plan grounded in the current goal, not the past one.

Respond with ONLY a JSON array of strings, nothing else. Example:
["Search for X", "Search for Y pricing", "Look up Z official documentation"]
"""


def _extract_json_array(text: str) -> list:
    """Best-effort extraction of a JSON array from an LLM response."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in planner output: {text!r}")
    return json.loads(match.group(0))


def plan(state: AgentState) -> dict:
    """Generate a research plan for state.user_goal. Returns a state update dict."""
    llm = get_llm_client()
    memory_hint = get_planning_hint(state.user_goal)
    prompt = f"Research goal: {state.user_goal}"
    if memory_hint:
        prompt += f"\n\n{memory_hint}"
    raw = llm.complete(prompt, system=PLANNER_SYSTEM_PROMPT, max_tokens=500)

    steps = _extract_json_array(raw)
    steps = [str(s).strip() for s in steps if str(s).strip()]

    if not steps:
        raise ValueError("Planner produced an empty plan.")

    print(f"[PLAN] Created {len(steps)} research step(s): {steps}")
    return {"plan": steps, "current_step": 0}
