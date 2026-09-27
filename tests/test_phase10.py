"""
Phase 10 — demo experience tests.

Covers the two things Phase 10 actually changed:
  1. app/main.py's `--demo` flag selects the built-in DEMO_QUESTION and
     prints clear USER TASK / FINAL REPORT / SUMMARY banners around the
     same underlying agent run (no behavior change to the graph itself).
  2. app/researcher.py now prints an explicit `[STATE]` line after each
     research pass, so the STATE UPDATE stage in the target workflow is
     visible in the terminal, not just implied.
"""

from __future__ import annotations

import sys

import app.main as main_module
from app.guardrails import validate_task
from app.state import AgentState, Source
from app.researcher import research


def test_demo_question_passes_validation():
    # The built-in demo question must itself be a valid task, or --demo
    # would fail guardrail validation before the agent ever runs.
    cleaned = validate_task(main_module.DEMO_QUESTION)
    assert cleaned == main_module.DEMO_QUESTION.strip()


def test_main_demo_flag_uses_demo_question_and_prints_banners(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["app.main", "--demo"])

    captured_goal = {}

    def fake_run(user_goal):
        captured_goal["goal"] = user_goal
        return AgentState(
            user_goal=user_goal,
            plan=["step 1"],
            completed_steps=["step 1"],
            sources=[Source(url="https://example.com", title="T", snippet="s")],
            iteration=1,
            final_report="# Research Report\n\nSome report body.",
        )

    monkeypatch.setattr(main_module, "run", fake_run)
    monkeypatch.setattr(
        main_module, "write_report", lambda goal, report: tmp_path / "report.md"
    )

    main_module.main()

    out = capsys.readouterr().out
    assert captured_goal["goal"] == main_module.DEMO_QUESTION
    assert "RESEARCHPILOT DEMO" in out
    assert "USER TASK" in out
    assert main_module.DEMO_QUESTION in out
    assert "FINAL REPORT" in out
    assert "Some report body." in out
    assert "SUMMARY" in out
    assert "[SAVED]" in out


def test_main_without_demo_flag_uses_provided_argument(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["app.main", "What", "is", "Python?"])

    def fake_run(user_goal):
        return AgentState(user_goal=user_goal, final_report="# Research Report\n\nBody.")

    monkeypatch.setattr(main_module, "run", fake_run)
    monkeypatch.setattr(
        main_module, "write_report", lambda goal, report: tmp_path / "report.md"
    )

    main_module.main()

    out = capsys.readouterr().out
    # Demo banner must NOT appear when --demo wasn't passed.
    assert "RESEARCHPILOT DEMO" not in out
    assert "What is Python?" in out


def test_main_exits_cleanly_when_agent_run_raises(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["app.main", "A valid question?"])

    def fake_run(user_goal):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "run", fake_run)

    try:
        main_module.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1

    out = capsys.readouterr().out
    assert "[ERROR]" in out
    assert "boom" in out


def test_research_prints_state_update_line(monkeypatch, capsys):
    def fake_web_search(query, max_results=3, timeout=10):
        from app.tools.web_search import SearchResult
        return [SearchResult(url="https://example.com/1", title="R", snippet="s")]

    import app.researcher as researcher_module
    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    state = AgentState(user_goal="test goal", plan=["Search something"])
    result = research(state)

    out = capsys.readouterr().out
    assert "[STATE]" in out
    assert "findings=1" in out
    assert "sources=1" in out
    assert "completed_steps=1/1" in out
    assert result["completed_steps"] == ["Search something"]
