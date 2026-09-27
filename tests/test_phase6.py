"""
Tests for Phase 6 (lightweight persistent memory).

Same approach as the earlier phase test files: no real network or LLM
calls. All memory tests use a temp-directory JSON path (via pytest's
`tmp_path`) instead of the real `memory/agent_memory.json`, so they never
touch (or depend on) this repo's actual memory file, and can freely
assert on exact file contents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import AgentState, Source, ToolCallRecord
from app.memory import (
    load_memory,
    save_memory,
    record_run,
    retrieve_relevant_experience,
    format_memory_hint,
    get_planning_hint,
    _tokenize,
)
import app.planner as planner_module


# ---------------------------------------------------------------------------
# load_memory / save_memory
# ---------------------------------------------------------------------------

def test_load_memory_returns_empty_store_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert load_memory(path) == {"runs": []}


def test_load_memory_returns_empty_store_on_corrupt_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_memory(path) == {"runs": []}


def test_save_then_load_roundtrips_and_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "mem.json"
    data = {"runs": [{"goal": "x", "successful_queries": []}]}
    save_memory(data, path)
    assert path.exists()
    assert load_memory(path) == data


# ---------------------------------------------------------------------------
# record_run
# ---------------------------------------------------------------------------

def test_record_run_splits_successful_and_failed_queries_and_persists(tmp_path):
    path = tmp_path / "mem.json"
    state = AgentState(
        user_goal="What is the pricing for Acme Cloud?",
        iteration=2,
        tool_history=[
            ToolCallRecord(tool="web_search", input="Acme Cloud pricing", success=True, summary="3 result(s)"),
            ToolCallRecord(tool="web_search", input="Acme Cloud xyzzy nonsense", success=True, summary="0 results"),
            ToolCallRecord(tool="web_search", input="Acme Cloud downtime", success=False, summary="network error"),
            ToolCallRecord(tool="calculator", input="10*12", success=True, summary="120"),
        ],
        sources=[
            Source(url="https://acme.com/pricing", title="Acme Pricing"),
            Source(url="https://acme.com/pricing#faq", title="Acme Pricing FAQ"),
            Source(url="https://blog.example.com/acme-review", title="Review"),
        ],
        critique={"sufficient": True, "missing_information": [], "issues": []},
    )

    record = record_run(state, path)

    assert record["goal"] == state.user_goal
    assert record["successful_queries"] == ["Acme Cloud pricing"]
    assert set(record["failed_queries"]) == {"Acme Cloud xyzzy nonsense", "Acme Cloud downtime"}
    # Calculator calls are not web searches and must not show up here.
    assert "10*12" not in record["successful_queries"] + record["failed_queries"]
    # Domains deduplicated (both acme.com URLs collapse to one entry).
    assert record["useful_domains"] == ["acme.com", "blog.example.com"]
    assert record["iterations"] == 2
    assert record["sufficient"] is True
    assert "timestamp" in record

    # Persisted to disk, not just returned.
    on_disk = load_memory(path)
    assert on_disk["runs"] == [record]


def test_record_run_appends_to_existing_runs(tmp_path):
    path = tmp_path / "mem.json"
    save_memory({"runs": [{"goal": "old run", "successful_queries": []}]}, path)

    state = AgentState(user_goal="new run", tool_history=[], sources=[])
    record_run(state, path)

    data = load_memory(path)
    assert len(data["runs"]) == 2
    assert data["runs"][0]["goal"] == "old run"
    assert data["runs"][1]["goal"] == "new run"


def test_record_run_defaults_sufficient_true_when_no_critique(tmp_path):
    path = tmp_path / "mem.json"
    state = AgentState(user_goal="g", tool_history=[], sources=[], critique=None)
    record = record_run(state, path)
    assert record["sufficient"] is True


# ---------------------------------------------------------------------------
# retrieve_relevant_experience
# ---------------------------------------------------------------------------

def _seed(path, goals):
    save_memory({"runs": [{"goal": g, "successful_queries": [], "failed_queries": [], "useful_domains": []} for g in goals]}, path)


def test_retrieve_relevant_experience_matches_on_keyword_overlap(tmp_path):
    path = tmp_path / "mem.json"
    _seed(path, [
        "What is the pricing for Notion?",
        "Best hiking trails near Seattle",
        "Notion vs Coda pricing comparison",
    ])

    relevant = retrieve_relevant_experience("How much does Notion cost per month?", path)

    goals = [r["goal"] for r in relevant]
    assert "Best hiking trails near Seattle" not in goals
    assert any("Notion" in g for g in goals)


def test_retrieve_relevant_experience_returns_empty_when_nothing_overlaps(tmp_path):
    path = tmp_path / "mem.json"
    _seed(path, ["Best hiking trails near Seattle"])
    assert retrieve_relevant_experience("PostgreSQL vs MySQL for OLTP workloads", path) == []


def test_retrieve_relevant_experience_empty_store(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert retrieve_relevant_experience("anything", path) == []


def test_retrieve_relevant_experience_respects_max_runs(tmp_path):
    path = tmp_path / "mem.json"
    _seed(path, [f"Notion pricing detail {i}" for i in range(5)])
    relevant = retrieve_relevant_experience("Notion pricing", path, max_runs=2)
    assert len(relevant) == 2


def test_tokenize_strips_stopwords_and_short_tokens():
    tokens = _tokenize("What is the pricing for Notion?")
    assert "pricing" in tokens
    assert "notion" in tokens
    assert "the" not in tokens
    assert "is" not in tokens
    assert "for" not in tokens


# ---------------------------------------------------------------------------
# format_memory_hint / get_planning_hint
# ---------------------------------------------------------------------------

def test_format_memory_hint_empty_for_no_runs():
    assert format_memory_hint([]) == ""


def test_format_memory_hint_includes_queries_and_domains():
    runs = [{
        "goal": "Notion pricing",
        "successful_queries": ["Notion official pricing page"],
        "failed_queries": ["Notion cost reddit"],
        "useful_domains": ["notion.so"],
    }]
    hint = format_memory_hint(runs)
    assert "Notion official pricing page" in hint
    assert "Notion cost reddit" in hint
    assert "notion.so" in hint


def test_get_planning_hint_returns_text_for_related_goal(tmp_path):
    path = tmp_path / "mem.json"
    save_memory({"runs": [{
        "goal": "Notion pricing",
        "successful_queries": ["Notion official pricing page"],
        "failed_queries": [],
        "useful_domains": ["notion.so"],
    }]}, path)

    hint = get_planning_hint("How much does the Notion paid plan cost?", path)
    assert "Notion official pricing page" in hint


def test_get_planning_hint_empty_string_when_nothing_relevant(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert get_planning_hint("Something totally unrelated", path) == ""


# ---------------------------------------------------------------------------
# planner integration: the memory hint reaches the LLM prompt
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    def complete(self, prompt, system="", max_tokens=500):
        self.last_prompt = prompt
        return self._text


def test_plan_includes_memory_hint_in_prompt_when_available(monkeypatch, tmp_path):
    path = tmp_path / "mem.json"
    save_memory({"runs": [{
        "goal": "Notion pricing",
        "successful_queries": ["Notion official pricing page"],
        "failed_queries": ["Notion cost reddit"],
        "useful_domains": ["notion.so"],
    }]}, path)

    monkeypatch.setenv("RESEARCHPILOT_MEMORY_PATH", str(path))
    fake = FakeLLM('["Search Notion pricing"]')
    monkeypatch.setattr(planner_module, "get_llm_client", lambda: fake)

    state = AgentState(user_goal="What does the Notion paid plan cost?")
    result = planner_module.plan(state)

    assert result["plan"] == ["Search Notion pricing"]
    assert "Notion official pricing page" in fake.last_prompt
    assert "Notion cost reddit" in fake.last_prompt


def test_plan_prompt_unchanged_when_no_relevant_memory(monkeypatch, tmp_path):
    path = tmp_path / "does_not_exist.json"
    monkeypatch.setenv("RESEARCHPILOT_MEMORY_PATH", str(path))
    fake = FakeLLM('["Search X"]')
    monkeypatch.setattr(planner_module, "get_llm_client", lambda: fake)

    state = AgentState(user_goal="Completely novel research goal")
    planner_module.plan(state)

    assert fake.last_prompt == f"Research goal: {state.user_goal}"


# ---------------------------------------------------------------------------
# persistence across "runs" (simulating separate process invocations)
# ---------------------------------------------------------------------------

def test_memory_persists_across_separate_record_and_retrieve_calls(tmp_path):
    path = tmp_path / "mem.json"

    # "Run 1": agent finishes a task and records it.
    state1 = AgentState(
        user_goal="Notion pricing for teams",
        iteration=1,
        tool_history=[ToolCallRecord(tool="web_search", input="Notion team pricing", success=True, summary="2 result(s)")],
        sources=[Source(url="https://notion.so/pricing", title="Pricing")],
    )
    record_run(state1, path)

    # Simulate a brand-new process: nothing in memory except what's on disk.
    relevant = retrieve_relevant_experience("How much is Notion for a team?", path)
    assert len(relevant) == 1
    assert relevant[0]["goal"] == "Notion pricing for teams"
    assert relevant[0]["successful_queries"] == ["Notion team pricing"]

    # "Run 2": a second, unrelated run also gets appended without losing run 1.
    state2 = AgentState(user_goal="Best espresso machines under $500", tool_history=[], sources=[])
    record_run(state2, path)

    data = load_memory(path)
    assert len(data["runs"]) == 2
    assert {r["goal"] for r in data["runs"]} == {
        "Notion pricing for teams",
        "Best espresso machines under $500",
    }
