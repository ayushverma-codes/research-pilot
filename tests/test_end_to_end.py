"""
Phase 9 — end-to-end workflow test.

Every other test file exercises one node/module in isolation with a fake
LLM. This file builds the real compiled LangGraph (app.graph.build_graph)
and drives a full run through it — planner -> researcher -> evaluator ->
(back to researcher once) -> reporter -> memory_writer -> END — with only
the external LLM calls and the network-touching web_search tool faked out.
Nothing about the graph wiring, routing function, or node contracts is
mocked: this is the same object app/main.py builds and invokes.

The scenario is deliberately built to require one extra research
iteration (evaluator says "insufficient" once, then "sufficient"), so the
conditional loop itself — not just the linear happy path — is verified.
"""

from __future__ import annotations

import json

import app.planner as planner_module
import app.evaluator as evaluator_module
import app.reporter as reporter_module
import app.researcher as researcher_module
import app.memory as memory_module

from app.graph import build_graph
from app.state import AgentState
from app.tools.web_search import SearchResult


class FakeLLM:
    """Returns queued responses in order; repeats the last one if exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, prompt, system="", max_tokens=500):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return self._responses[-1] if self._responses else ""


def test_full_graph_run_with_one_extra_research_iteration(monkeypatch, tmp_path):
    # Isolate the memory store so this test can't read/write the real
    # project's memory/agent_memory.json.
    monkeypatch.setenv("RESEARCHPILOT_MEMORY_PATH", str(tmp_path / "mem.json"))

    # --- planner: a single-step plan, so a second research pass can only
    # happen if the critic actively queues a genuinely new query ---
    planner_llm = FakeLLM(['["Search Acme pricing"]'])
    monkeypatch.setattr(planner_module, "get_llm_client", lambda: planner_llm)

    # --- researcher: fake web_search so no real network call is made ---
    search_calls = []

    def fake_web_search(query, max_results=3, timeout=10):
        search_calls.append(query)
        return [
            SearchResult(
                url=f"https://example.com/{len(search_calls)}",
                title=f"Result for {query}",
                snippet=f"Some snippet about {query}",
            )
        ]

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    # --- evaluator/critic: insufficient on pass 1, sufficient on pass 2 ---
    evaluator_llm = FakeLLM([
        json.dumps({
            "sufficient": False,
            "missing_information": ["Official pricing page has not been checked yet"],
            "issues": ["Only one informal source found so far"],
            "recommended_action": "Search the official pricing page directly",
            "additional_queries": ["Search Acme official pricing page"],
        }),
        json.dumps({
            "sufficient": True,
            "missing_information": [],
            "issues": [],
            "recommended_action": "Proceed to report",
        }),
    ])
    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: evaluator_llm)

    # --- reporter: well-formed four-section synthesis ---
    reporter_llm = FakeLLM(["""## Executive Summary
Acme's pricing has a free tier and a paid tier based on gathered findings.

## Key Findings
- Acme offers a free tier.
- Acme's paid tier is listed on its official pricing page.

## Comparison / Analysis
The two sources are consistent with each other on the existence of a paid tier.

## Limitations
- Exact paid tier price was not captured in the findings.
"""])
    monkeypatch.setattr(reporter_module, "get_llm_client", lambda: reporter_llm)

    graph = build_graph()
    initial_state = AgentState(user_goal="What is Acme's pricing?")
    result = graph.invoke(initial_state)
    final_state = AgentState(**result)

    # --- the loop actually looped: researcher ran for both plan steps,
    # across two evaluator passes, not a single fixed pass ---
    assert search_calls == ["Search Acme pricing", "Search Acme official pricing page"]
    assert evaluator_llm.calls == 2
    assert final_state.iteration == 2
    assert final_state.critique["sufficient"] is True

    # --- state accumulated correctly across nodes ---
    assert set(final_state.completed_steps) == {
        "Search Acme pricing",
        "Search Acme official pricing page",
    }
    assert len(final_state.sources) == 2
    assert all(record.tool == "web_search" for record in final_state.tool_history)

    # --- reporter produced the full structured report, not a fallback ---
    assert "# Research Report" in final_state.final_report
    assert "## Executive Summary" in final_state.final_report
    assert "## Sources" in final_state.final_report
    assert "example.com" in final_state.final_report

    # --- memory_writer actually persisted this run for future planning ---
    persisted = memory_module.load_memory(tmp_path / "mem.json")
    assert len(persisted["runs"]) == 1
    assert persisted["runs"][0]["goal"] == "What is Acme's pricing?"


def test_full_graph_run_stops_at_max_iterations_without_looping_forever(monkeypatch, tmp_path):
    """A critic that never says 'sufficient' must not spin past MAX_ITERATIONS."""
    monkeypatch.setenv("RESEARCHPILOT_MEMORY_PATH", str(tmp_path / "mem.json"))
    monkeypatch.setattr(evaluator_module, "MAX_ITERATIONS", 2)

    planner_llm = FakeLLM(['["Search vague topic"]'])
    monkeypatch.setattr(planner_module, "get_llm_client", lambda: planner_llm)

    def fake_web_search(query, max_results=3, timeout=10):
        return [SearchResult(url="https://example.com/1", title="R", snippet="s")]

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    # Always claims more research is needed and always proposes a fresh
    # query, so the loop would run forever without the hard iteration cap.
    # The router must still bail out once MAX_ITERATIONS is reached, per
    # app/evaluator.py's own iteration-limit short-circuit (which stops
    # calling the LLM at all once the cap is hit).
    def _always_insufficient(prompt, system="", max_tokens=500):
        n = _always_insufficient.calls
        _always_insufficient.calls += 1
        return json.dumps({
            "sufficient": False,
            "missing_information": ["Still missing something"],
            "issues": [],
            "recommended_action": "Keep searching",
            "additional_queries": [f"Search vague topic again #{n}"],
        })

    _always_insufficient.calls = 0

    class _AlwaysInsufficientLLM:
        def complete(self, prompt, system="", max_tokens=500):
            return _always_insufficient(prompt, system, max_tokens)

    evaluator_llm = _AlwaysInsufficientLLM()
    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: evaluator_llm)

    reporter_llm = FakeLLM(["""## Executive Summary
Findings were limited.

## Key Findings
- Only partial information was found.

## Comparison / Analysis
Not enough evidence for a full comparison.

## Limitations
- Evidence sufficiency was never confirmed.
"""])
    monkeypatch.setattr(reporter_module, "get_llm_client", lambda: reporter_llm)

    graph = build_graph()
    result = graph.invoke(AgentState(user_goal="Vague ambiguous goal"))
    final_state = AgentState(**result)

    # iteration is incremented before the max-iterations check fires, so the
    # loop is bounded at MAX_ITERATIONS + 1 passes, never unbounded.
    assert final_state.iteration == 3
    assert evaluator_module.MAX_ITERATIONS == 2
    assert final_state.critique["sufficient"] is True  # forced sufficient once cap is hit
    assert final_state.final_report  # a report was still produced, not an empty string
    assert "# Research Report" in final_state.final_report
