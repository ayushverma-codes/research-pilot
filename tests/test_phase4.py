"""
Tests for Phase 4 (the critic's quality checks on top of Phase 2's
coverage check).

Same approach as test_phase2.py / test_phase3.py: no real network or LLM
calls. `evaluate_evidence` is exercised with a FakeLLM that returns
canned JSON, so we can check that the new "issues" / "recommended_action"
fields are parsed and surfaced in `state.critique`, without touching the
coverage/routing behaviour Phase 2/3 already cover and test.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import AgentState, Source
import app.evaluator as evaluator_module
from app.evaluator import evaluate_evidence, MAX_ITERATIONS


class FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt, system="", max_tokens=700):
        return json.dumps(self._payload)


def test_critique_surfaces_issues_even_when_sufficient(monkeypatch):
    # Coverage is fine (sufficient=true, no additional_queries), but the
    # critic still flags a quality issue - this must show up in
    # state.critique without forcing another research pass.
    monkeypatch.setattr(
        evaluator_module,
        "get_llm_client",
        lambda: FakeLLM({
            "sufficient": True,
            "missing_information": [],
            "issues": ["One claim about pricing has no source behind it"],
            "recommended_action": "Proceed to report, but caveat the unsupported claim.",
            "additional_queries": [],
        }),
    )

    state = AgentState(
        user_goal="What does X cost?",
        findings=["Step 'search': X costs $10/mo (no source given)"],
        sources=[Source(url="http://real.com/pricing", title="Real pricing page")],
    )
    update = evaluate_evidence(state)

    assert update["missing_information"] == []
    assert "critique" in update
    assert update["critique"]["sufficient"] is True
    assert "no source" in update["critique"]["issues"][0]
    assert "caveat" in update["critique"]["recommended_action"]


def test_critique_populated_when_more_research_needed(monkeypatch):
    monkeypatch.setattr(
        evaluator_module,
        "get_llm_client",
        lambda: FakeLLM({
            "sufficient": False,
            "missing_information": ["exact pricing"],
            "issues": [],
            "recommended_action": "Search the official pricing page.",
            "additional_queries": ["official X pricing page"],
        }),
    )

    state = AgentState(user_goal="What does X cost?", findings=["some finding"])
    update = evaluate_evidence(state)

    assert update["missing_information"] == ["exact pricing"]
    assert update["critique"]["sufficient"] is False
    assert update["critique"]["recommended_action"] == "Search the official pricing page."
    assert "official X pricing page" in update["plan"]


def test_critique_present_at_max_iterations_without_calling_llm():
    state = AgentState(user_goal="g", iteration=MAX_ITERATIONS, findings=["some finding"])
    update = evaluate_evidence(state)

    assert update["iteration"] == MAX_ITERATIONS + 1
    assert update["missing_information"] == []
    assert update["critique"]["sufficient"] is True
    assert "Max iterations" in update["critique"]["recommended_action"]


def test_critique_present_when_llm_output_unparsable(monkeypatch):
    monkeypatch.setattr(
        evaluator_module,
        "get_llm_client",
        lambda: FakeLLM_broken(),
    )

    state = AgentState(user_goal="g", findings=["some finding"])
    update = evaluate_evidence(state)

    assert update["missing_information"] == []
    assert update["critique"]["sufficient"] is True
    assert update["critique"]["issues"]  # the failure itself is recorded as an issue


class FakeLLM_broken:
    def complete(self, prompt, system="", max_tokens=700):
        return "not json at all"


def test_evaluate_evidence_retries_once_after_empty_response(monkeypatch):
    # Regression test for a live failure: the LLM returned an empty string
    # on one call, which the old code treated as a final failure straight
    # away. It should now retry once and succeed on the second attempt
    # instead of silently assuming "sufficient".
    calls = []

    class FlakyThenGoodLLM:
        def complete(self, prompt, system="", max_tokens=700):
            calls.append(prompt)
            if len(calls) == 1:
                return ""  # simulates the empty response seen live
            return json.dumps({
                "sufficient": False,
                "missing_information": ["exact pricing"],
                "issues": [],
                "recommended_action": "Search the official pricing page.",
                "additional_queries": ["official pricing page"],
            })

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: FlakyThenGoodLLM())

    state = AgentState(user_goal="What does X cost?", findings=["some finding"])
    update = evaluate_evidence(state)

    assert len(calls) == 2
    assert "official pricing page" in update["plan"]
    assert update["critique"]["sufficient"] is False
    # The failure never happened from the router's point of view -
    # no "evaluation failed" issue should be recorded when the retry works.
    assert not any("failed" in issue.lower() for issue in update["critique"]["issues"])


def test_evaluate_evidence_falls_back_after_two_failed_attempts(monkeypatch):
    calls = []

    class AlwaysBrokenLLM:
        def complete(self, prompt, system="", max_tokens=700):
            calls.append(prompt)
            return ""

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: AlwaysBrokenLLM())

    state = AgentState(user_goal="g", findings=["some finding"])
    update = evaluate_evidence(state)

    assert len(calls) == 2  # both attempts were used, not just one
    assert update["critique"]["sufficient"] is True
    assert "2 attempt" in update["critique"]["issues"][0]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
