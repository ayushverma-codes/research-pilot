"""
Basic tests for Phase 1 MVP.

These tests avoid requiring network access or an LLM API key, so they can
run anywhere. Full end-to-end testing (planner + researcher + reporter with
live LLM and live web search) is done manually via `python -m app.main`,
as documented in the README, since it requires real credentials/network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import AgentState, Source, ToolCallRecord
from app.tools.calculator import calculate, CalculatorError
from app.planner import _extract_json_array
from app.llm_provider import RateLimiter, GroqClient


def test_agent_state_defaults():
    state = AgentState(user_goal="test goal")
    assert state.user_goal == "test goal"
    assert state.plan == []
    assert state.current_step == 0
    assert state.findings == []
    assert state.iteration == 0
    assert state.final_report == ""


def test_agent_state_accepts_updates():
    state = AgentState(user_goal="g")
    state.sources.append(Source(url="http://x.com", title="X", snippet="y"))
    state.tool_history.append(
        ToolCallRecord(tool="web_search", input="q", success=True, summary="1 result")
    )
    assert len(state.sources) == 1
    assert state.tool_history[0].tool == "web_search"


def test_calculator_basic_ops():
    assert calculate("2 + 3") == 5
    assert calculate("10 / 4") == 2.5
    assert calculate("2 ** 5") == 32
    assert calculate("-3 + 7") == 4


def test_calculator_rejects_unsafe_input():
    import pytest
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('echo hi')")


def test_extract_json_array_from_planner_style_text():
    raw = 'Sure, here is the plan:\n["step one", "step two", "step three"]'
    steps = _extract_json_array(raw)
    assert steps == ["step one", "step two", "step three"]


def test_extract_json_array_raises_on_no_array():
    import pytest
    with pytest.raises(ValueError):
        _extract_json_array("no array here")


def test_rate_limiter_enforces_min_interval():
    import time as time_module

    limiter = RateLimiter(requests_per_minute=600)  # min_interval = 0.1s
    start = time_module.monotonic()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time_module.monotonic() - start
    # 3 calls at 600 rpm => at least 2 * 0.1s gaps enforced
    assert elapsed >= 0.19


def test_groq_client_parses_retry_after_header():
    class FakeHeaders:
        def get(self, key):
            return "2.5" if key == "retry-after" else None

    class FakeResponse:
        headers = FakeHeaders()

    class FakeError(Exception):
        response = FakeResponse()

    delay = GroqClient._parse_retry_after(FakeError(), default=9.0)
    assert delay == 2.5


def test_groq_client_falls_back_to_default_when_no_header():
    class FakeError(Exception):
        pass

    delay = GroqClient._parse_retry_after(FakeError(), default=7.0)
    assert delay == 7.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
