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
    assert update["critique"]["sufficient"] is False
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
    assert update["critique"]["sufficient"] is False
    assert "2 attempt" in update["critique"]["issues"][0]


def test_evaluate_evidence_bounds_error_message_for_truncated_json(monkeypatch):
    # Regression test for a live failure: a long response cut off by
    # max_tokens before its JSON closed produced an error whose message
    # embedded the model's *entire* raw reply verbatim (via repr), which
    # then flowed into state.critique["issues"] and from there into the
    # final report. The error message must stay short regardless of how
    # long the truncated raw reply was.
    long_truncated_json = (
        '{\n  "sufficient": false,\n  "missing_information": [\n    "'
        + ("very long detail that keeps going " * 50)
        + '"\n  ]'
        # deliberately no closing "}" - simulates a max_tokens cutoff
    )

    class TruncatedLLM:
        def complete(self, prompt, system="", max_tokens=700):
            return long_truncated_json

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: TruncatedLLM())

    state = AgentState(user_goal="g", findings=["some finding"])
    update = evaluate_evidence(state)

    assert update["critique"]["sufficient"] is False
    issue = update["critique"]["issues"][0]
    assert len(issue) < 400  # bounded, not the ~1800-char raw reply
    assert "\n" not in issue


def test_evaluator_uses_a_bounded_json_token_budget():
    # The critic only emits a small JSON object. Keep enough headroom to avoid
    # truncation while preventing later evidence checks from requesting an
    # unnecessarily large generation.
    assert 700 <= evaluator_module.EVALUATOR_MAX_TOKENS <= 1000


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def test_critic_compacts_large_findings_prompt(monkeypatch):
    seen_prompts = []

    class InspectingLLM:
        def complete(self, prompt, system="", max_tokens=700):
            seen_prompts.append(prompt)
            return json.dumps({
                "sufficient": True,
                "missing_information": [],
                "issues": [],
                "recommended_action": "Proceed to report.",
                "additional_queries": [],
            })

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: InspectingLLM())
    state = AgentState(user_goal="g", findings=["x" * 5000 for _ in range(10)])
    update = evaluate_evidence(state)

    assert update["critique"]["sufficient"] is True
    assert len(seen_prompts) == 1
    assert len(seen_prompts[0]) < evaluator_module.EVALUATOR_MAX_FINDINGS_CHARS + 2000
    assert "Earlier findings omitted" in seen_prompts[0]


def test_evaluator_timeout_queues_one_deeper_source_pass_without_second_wait(monkeypatch):
    import time

    class StuckLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, system="", max_tokens=700):
            self.calls += 1
            time.sleep(0.5)
            return "{}"

    stuck = StuckLLM()
    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: stuck)
    monkeypatch.setattr(evaluator_module, "EVALUATOR_CALL_TIMEOUT_SECONDS", 0.05)

    state = AgentState(
        user_goal="What does X cost?",
        findings=["some finding"],
        sources=[Source(url="https://example.com/pricing", title="X pricing")],
    )
    started = time.monotonic()
    update = evaluate_evidence(state)
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert stuck.calls == 1
    assert update["missing_information"]
    assert update["critique"]["sufficient"] is False
    assert "https://example.com/pricing" in update["plan"]
    assert "1 attempt" in update["critique"]["issues"][0]


def test_evaluator_second_timeout_stops_after_bounded_recovery(monkeypatch):
    import time

    class StuckLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, system="", max_tokens=700):
            self.calls += 1
            time.sleep(0.5)
            return "{}"

    stuck = StuckLLM()
    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: stuck)
    monkeypatch.setattr(evaluator_module, "EVALUATOR_CALL_TIMEOUT_SECONDS", 0.05)

    # iteration=1 means this is the critic pass after the one-shot recovery.
    state = AgentState(
        user_goal="What does X cost?",
        iteration=1,
        findings=["detailed page finding"],
        sources=[Source(url="https://example.com/pricing", title="X pricing")],
        completed_steps=["https://example.com/pricing"],
    )
    update = evaluate_evidence(state)

    assert stuck.calls == 1
    assert update["missing_information"] == []
    assert update["critique"]["sufficient"] is False
    assert "could not be confirmed" in update["critique"]["issues"][-1].lower()
    assert "plan" not in update


def test_timeout_recovery_can_read_source_then_reach_sufficient_critic(monkeypatch):
    from app.researcher import research
    import app.researcher as researcher_module
    from app.tools.page_reader import PageContent

    # Start from the state that exists after the initial searches: a relevant
    # URL is known, but only snippet-level evidence has been gathered.
    state = AgentState(
        user_goal="What does X cost?",
        plan=["search X pricing"],
        completed_steps=["search X pricing"],
        findings=["Search result says the official pricing page exists."],
        sources=[Source(url="https://example.com/pricing", title="Official X pricing", snippet="Pricing")],
    )

    class StuckLLM:
        def complete(self, prompt, system="", max_tokens=700):
            time.sleep(0.2)
            return "{}"

    import time
    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: StuckLLM())
    monkeypatch.setattr(evaluator_module, "EVALUATOR_CALL_TIMEOUT_SECONDS", 0.02)
    first_update = evaluate_evidence(state)
    state = state.model_copy(update=first_update)

    assert evaluator_module.needs_more_research(state) == "researcher"
    assert "https://example.com/pricing" in state.plan

    monkeypatch.setattr(
        researcher_module,
        "read_page",
        lambda url: PageContent(url=url, title="Official X pricing", text="X costs $10 per month."),
    )
    research_update = research(state)
    state = state.model_copy(update=research_update)
    assert state.tool_history[-1].tool == "page_reader"

    class GoodLLM:
        def complete(self, prompt, system="", max_tokens=700):
            return json.dumps({
                "sufficient": True,
                "missing_information": [],
                "issues": [],
                "recommended_action": "Evidence is sufficient and ready for the report.",
                "additional_queries": [],
            })

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: GoodLLM())
    monkeypatch.setattr(evaluator_module, "_run_with_hard_timeout", lambda call, timeout: call())
    second_update = evaluate_evidence(state)
    state = state.model_copy(update=second_update)

    assert state.iteration == 2
    assert state.critique["sufficient"] is True
    assert evaluator_module.needs_more_research(state) == "reporter"
