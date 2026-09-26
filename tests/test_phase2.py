"""
Tests for Phase 2 (the conditional research/evaluate loop).

Like test_mvp.py, these avoid real network access or an LLM API key:
- `_extract_json_object` is tested directly (pure function).
- `needs_more_research` (the router) is tested directly against
  hand-built AgentState instances.
- `evaluate_evidence`'s max-iteration cutoff is tested directly, since that
  branch returns before ever calling the LLM client.
- The researcher's "don't re-search completed steps" dedup logic is tested
  by monkeypatching `app.researcher.web_search` so no real HTTP call is
  made.

Full live behaviour (LLM actually deciding to loop) is verified manually
via `python -m app.main "..."`, same as Phase 1, since that requires real
credentials and network access.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import AgentState, Source
from app.evaluator import _extract_json_object, needs_more_research, evaluate_evidence, MAX_ITERATIONS
import app.researcher as researcher_module


def test_extract_json_object_from_evaluator_style_text():
    raw = 'Here you go:\n{"sufficient": false, "missing_information": ["pricing"], "additional_queries": ["X official pricing"]}'
    result = _extract_json_object(raw)
    assert result["sufficient"] is False
    assert result["missing_information"] == ["pricing"]
    assert result["additional_queries"] == ["X official pricing"]


def test_extract_json_object_raises_on_no_object():
    import pytest
    with pytest.raises(ValueError):
        _extract_json_object("no json here")


def test_router_sends_back_to_researcher_when_missing_info_and_under_limit():
    state = AgentState(
        user_goal="g",
        missing_information=["pricing details"],
        iteration=1,
    )
    assert needs_more_research(state) == "researcher"


def test_router_sends_to_reporter_when_no_missing_info():
    state = AgentState(user_goal="g", missing_information=[], iteration=1)
    assert needs_more_research(state) == "reporter"


def test_router_sends_to_reporter_when_iteration_limit_hit_even_with_missing_info():
    # Guards against infinite loops: missing_information alone can't force
    # another pass once we're past MAX_ITERATIONS.
    state = AgentState(
        user_goal="g",
        missing_information=["still missing something"],
        iteration=MAX_ITERATIONS + 1,
    )
    assert needs_more_research(state) == "reporter"


def test_evaluate_evidence_stops_at_max_iterations_without_calling_llm():
    # iteration = MAX_ITERATIONS already -> next iteration would exceed the
    # cap, so this must short-circuit and NOT try to build an LLM client
    # (which would raise LLMError with no API key configured).
    state = AgentState(user_goal="g", iteration=MAX_ITERATIONS, findings=["some finding"])
    update = evaluate_evidence(state)
    assert update["iteration"] == MAX_ITERATIONS + 1
    assert update["missing_information"] == []


def test_researcher_skips_already_completed_steps(monkeypatch):
    calls = []

    def fake_web_search(query, max_results=3):
        calls.append(query)
        return []

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    state = AgentState(
        user_goal="g",
        plan=["step one", "step two", "step three"],
        completed_steps=["step one", "step two"],
    )
    update = researcher_module.research(state)

    # Only the new, not-yet-completed step should have triggered a search.
    assert calls == ["step three"]
    assert "step three" in update["completed_steps"]
    assert update["completed_steps"].count("step one") == 1


def test_researcher_records_sources_for_new_step(monkeypatch):
    from app.tools.web_search import SearchResult

    def fake_web_search(query, max_results=3):
        return [SearchResult(title="T", url="http://x.com", snippet="s")]

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    state = AgentState(user_goal="g", plan=["only step"])
    update = researcher_module.research(state)

    assert len(update["sources"]) == 1
    assert isinstance(update["sources"][0], Source)
    assert update["sources"][0].url == "http://x.com"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
