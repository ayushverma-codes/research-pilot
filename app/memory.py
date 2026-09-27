"""
Lightweight persistent memory (Phase 6).

Goal, explicitly: previous experience -> better planning/search strategy
for future runs. This is NOT model training and NOT a vector database -
just a small JSON file recording, per past run, what was searched for,
what worked, what failed, and which source domains turned out useful.
Before planning a new task, the planner retrieves entries whose goal text
overlaps with the new goal and folds a short summary of them into its
prompt as extra context (never as ground truth - the LLM still has to
produce a real plan).

Storage format (memory/agent_memory.json by default):

    {
      "runs": [
        {
          "timestamp": "2026-01-01T12:00:00",
          "goal": "...",
          "successful_queries": ["...", ...],
          "failed_queries": ["...", ...],
          "useful_domains": ["example.com", ...],
          "iterations": 2,
          "sufficient": true
        },
        ...
      ]
    }

JSON (not SQLite) because a single flat list of small run records is all
this needs - no relational structure, no concurrent-writer story to
solve. If the file grows large enough that this stops being true, that's
a reason to revisit, not a reason to build it up front.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.state import AgentState

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parent.parent / "memory" / "agent_memory.json"

# How many past runs' worth of hints to fold into a planner prompt. Kept
# small on purpose - this is a nudge ("here's what worked/failed before"),
# not a dump of the agent's entire history.
DEFAULT_MAX_RELEVANT_RUNS = 3

# Minimal stopword list for the keyword-overlap relevance match below.
# Deliberately tiny: this only needs to strip the most common connective
# words so "pricing" and "official" carry more weight than "the"/"a".
_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "is", "are",
    "what", "how", "does", "do", "with", "about", "vs", "between", "or",
    "be", "than", "that", "this", "it", "at", "as", "by",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    """Lowercase word tokens, minus stopwords/short noise tokens."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _memory_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)

    # Treat an unset *or blank* environment variable as "use the default".
    # Path("") resolves to the current directory ("."), which would make
    # load_memory try to read a directory as JSON and save_memory try to
    # overwrite it. A blank value is common in copied .env templates, so it
    # must be handled explicitly.
    configured = os.getenv("RESEARCHPILOT_MEMORY_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_MEMORY_PATH


def load_memory(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the memory store, or an empty one if it doesn't exist yet /
    can't be parsed. A missing or corrupt memory file must never break a
    run - it just means no past experience is available this time.
    """
    p = _memory_path(path)
    if not p.exists():
        return {"runs": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[MEMORY] Could not read memory file ({e}) — starting fresh.")
        return {"runs": []}
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        return {"runs": []}
    return data


def save_memory(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Persist the memory store to disk, creating the directory if needed."""
    p = _memory_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def _build_run_record(state: AgentState) -> Dict[str, Any]:
    """Turn a finished run's state into a memory record (see module docstring)."""
    successful_queries: List[str] = []
    failed_queries: List[str] = []
    for rec in state.tool_history:
        if rec.tool != "web_search":
            continue
        if rec.success and rec.summary != "0 results":
            successful_queries.append(rec.input)
        else:
            failed_queries.append(rec.input)

    useful_domains: List[str] = []
    seen_domains = set()
    for s in state.sources:
        domain = _extract_domain(s.url)
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            useful_domains.append(domain)

    sufficient = True
    if state.critique is not None:
        sufficient = bool(state.critique.get("sufficient", True))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "goal": state.user_goal,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "useful_domains": useful_domains,
        "iterations": state.iteration,
        "sufficient": sufficient,
    }


def record_run(state: AgentState, path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Append this run's experience to the memory store and save it.
    Returns the record that was written (mainly for tests/logging).
    """
    record = _build_run_record(state)
    data = load_memory(path)
    data["runs"].append(record)
    save_memory(data, path)
    return record


def retrieve_relevant_experience(
    user_goal: str,
    path: Optional[Path] = None,
    max_runs: int = DEFAULT_MAX_RELEVANT_RUNS,
) -> List[Dict[str, Any]]:
    """
    Return up to `max_runs` past run records whose goal shares at least
    one non-stopword keyword with `user_goal`, most-overlapping first.
    Simple word-overlap scoring - no embeddings/vector search, in
    keeping with "lightweight" and "do not over-engineer".
    """
    data = load_memory(path)
    runs = data.get("runs", [])
    if not runs:
        return []

    goal_tokens = _tokenize(user_goal)
    if not goal_tokens:
        return []

    scored = []
    for run in runs:
        past_tokens = _tokenize(run.get("goal", ""))
        overlap = goal_tokens & past_tokens
        if overlap:
            scored.append((len(overlap), run))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [run for _, run in scored[:max_runs]]


def format_memory_hint(relevant_runs: List[Dict[str, Any]]) -> str:
    """
    Render relevant past runs as a short block of plain-text hints for
    the planner prompt. Returns "" when there's nothing relevant, so
    callers can skip adding an empty section.
    """
    if not relevant_runs:
        return ""

    lines = [
        "Notes from related past research runs (use as hints only, not facts):"
    ]
    for run in relevant_runs:
        lines.append(f"- Past goal: {run.get('goal', '')}")
        if run.get("successful_queries"):
            lines.append(f"  Queries that worked: {', '.join(run['successful_queries'])}")
        if run.get("failed_queries"):
            lines.append(f"  Queries that failed/found nothing: {', '.join(run['failed_queries'])}")
        if run.get("useful_domains"):
            lines.append(f"  Useful source domains: {', '.join(run['useful_domains'])}")
    return "\n".join(lines)


def get_planning_hint(user_goal: str, path: Optional[Path] = None) -> str:
    """Convenience wrapper: retrieve + format in one call for planner.py."""
    relevant = retrieve_relevant_experience(user_goal, path=path)
    if relevant:
        print(f"[MEMORY] Found {len(relevant)} related past run(s) — adding hints to the plan prompt.")
    else:
        print("[MEMORY] No related past runs found.")
    return format_memory_hint(relevant)
