"""
Tests for Phase 5 (structured, multi-section report generation).

Same approach as test_phase2/3/4.py: no real network or LLM calls.
`generate_report` is exercised with a FakeLLM that returns canned section
markdown, so we can check that the deterministic sections (Research
Question, Methodology, Limitations base, Sources) are assembled correctly
from `state` and combined with the LLM's narrative sections, and that
`write_report` no longer double-wraps an already-formatted report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import AgentState, Source, ToolCallRecord
import app.reporter as reporter_module
from app.reporter import generate_report, _split_sections, _clean_bullet_text
from app.tools.report_writer import write_report


class FakeLLM:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt, system="", max_tokens=2000):
        return self._text


GOOD_SECTIONS = """## Executive Summary
X costs $10/month according to the official pricing page.

## Key Findings
- Official price is $10/month.
- Annual billing gives a discount.

## Comparison / Analysis
Compared to competitor Y at $15/month, X is the cheaper option.

## Limitations
- Enterprise/volume pricing was not found in the gathered sources.
"""


# ---------------------------------------------------------------------------
# _split_sections
# ---------------------------------------------------------------------------

def test_split_sections_parses_headings_in_order():
    sections = _split_sections(GOOD_SECTIONS)
    assert list(sections.keys()) == [
        "Executive Summary",
        "Key Findings",
        "Comparison / Analysis",
        "Limitations",
    ]
    assert "X costs $10/month" in sections["Executive Summary"]
    assert "Annual billing" in sections["Key Findings"]


def test_split_sections_returns_empty_dict_when_no_headings():
    assert _split_sections("just plain text, no markdown headers here") == {}


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def test_generate_report_produces_all_required_sections_in_order(monkeypatch):
    monkeypatch.setattr(reporter_module, "get_llm_client", lambda: FakeLLM(GOOD_SECTIONS))

    state = AgentState(
        user_goal="How much does X cost per month?",
        findings=["Step 'search X pricing': X costs $10/month, official site."],
        sources=[
            Source(url="https://x.com/pricing", title="X Pricing"),
            Source(url="https://x.com/pricing", title="X Pricing (dup)"),  # duplicate URL
        ],
        tool_history=[ToolCallRecord(tool="web_search", input="X pricing", success=True, summary="1 result")],
        iteration=1,
        critique={"sufficient": True, "issues": [], "recommended_action": "Evidence sufficient."},
    )

    update = generate_report(state)
    report = update["final_report"]

    # Sections present, in the required order.
    for heading in [
        "# Research Report",
        "## Executive Summary",
        "## Research Question",
        "## Methodology",
        "## Key Findings",
        "## Comparison / Analysis",
        "## Limitations",
        "## Sources",
    ]:
        assert heading in report
    assert report.index("## Executive Summary") < report.index("## Research Question")
    assert report.index("## Research Question") < report.index("## Methodology")
    assert report.index("## Methodology") < report.index("## Key Findings")
    assert report.index("## Key Findings") < report.index("## Comparison / Analysis")
    assert report.index("## Comparison / Analysis") < report.index("## Limitations")
    assert report.index("## Limitations") < report.index("## Sources")

    # Deterministic content actually reflects state, not the LLM.
    assert "How much does X cost per month?" in report
    assert "1 research iteration(s)" in report
    assert "web_search x1" in report
    assert "Evidence sufficient." in report

    # LLM narrative content made it through.
    assert "Annual billing gives a discount" in report
    assert "cheaper option" in report

    # Sources de-duplicated by URL.
    assert report.count("https://x.com/pricing") == 1


def test_generate_report_states_insufficiency_when_no_findings(monkeypatch):
    # Even if the LLM is well-behaved, a run with no findings/sources
    # should surface that fact deterministically in Limitations.
    monkeypatch.setattr(
        reporter_module,
        "get_llm_client",
        lambda: FakeLLM(
            "## Executive Summary\nNo evidence was found to answer this question.\n\n"
            "## Key Findings\n(none)\n\n## Comparison / Analysis\n(none)\n\n"
            "## Limitations\nNo findings were gathered at all.\n"
        ),
    )

    state = AgentState(user_goal="Obscure question with no results", findings=[], sources=[])
    update = generate_report(state)
    report = update["final_report"]

    assert "No sources were successfully gathered" in report
    assert "No evidence was found" in report


def test_generate_report_falls_back_when_llm_ignores_format(monkeypatch):
    # If the LLM doesn't use the requested headings, its whole reply
    # should still show up (as the executive summary) instead of being
    # silently dropped.
    monkeypatch.setattr(reporter_module, "get_llm_client", lambda: FakeLLM("Just a plain unstructured answer."))

    state = AgentState(user_goal="g", findings=["some finding"])
    update = generate_report(state)
    report = update["final_report"]

    assert "Just a plain unstructured answer." in report
    assert "(no findings gathered)" in report  # Key Findings section fallback


# ---------------------------------------------------------------------------
# write_report backward compatibility
# ---------------------------------------------------------------------------

def test_write_report_writes_already_formatted_report_as_is(tmp_path):
    content = "# Research Report\n\n## Executive Summary\n\nDone.\n"
    path = write_report("irrelevant goal", content, output_dir=tmp_path)

    text = path.read_text(encoding="utf-8")
    # No double "# Research Report" / no injected "**Goal:**" wrapper.
    assert text.count("# Research Report") == 1
    assert "**Goal:**" not in text
    assert text == content


def test_write_report_still_wraps_bare_content(tmp_path):
    # Backward compatibility with pre-Phase-5 callers (test_phase3.py).
    path = write_report("What is X?", "X is Y.", output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "# Research Report" in text
    assert "**Goal:** What is X?" in text
    assert "X is Y." in text


# ---------------------------------------------------------------------------
# _clean_bullet_text / limitations sanitization
#
# Regression coverage for a live failure: when the critic's own LLM call
# failed, the raw exception text (a long, multi-line, quote-heavy repr of
# the model's truncated JSON reply) was stored verbatim in
# state.critique["issues"] and then landed unmodified in the report's
# Limitations section - see evaluator.py's `_preview`/`_extract_json_object`
# and reporter.py's `_clean_bullet_text` for the fix on each side.
# ---------------------------------------------------------------------------

def test_clean_bullet_text_collapses_newlines_and_truncates():
    messy = "line one\nline two\n\n" + ("x" * 400)
    cleaned = _clean_bullet_text(messy)

    assert "\n" not in cleaned
    assert "line one line two" in cleaned
    assert len(cleaned) <= 220 + len("...(truncated)")
    assert cleaned.endswith("...(truncated)")


def test_generate_report_sanitizes_messy_critique_issue(monkeypatch):
    # Simulate a critic failure that stored a long, multi-line, JSON-ish
    # error message as an "issue" - the report must not reproduce it
    # verbatim.
    monkeypatch.setattr(reporter_module, "get_llm_client", lambda: FakeLLM(GOOD_SECTIONS))

    messy_issue = (
        "Critic evaluation itself failed after 2 attempt(s): "
        "Evaluator output was not valid JSON: '{\n  \"sufficient\": false,\n"
        "  \"missing_information\": [\n    \"" + ("very long detail " * 20) + "\"\n  ]"
    )
    state = AgentState(
        user_goal="g",
        findings=["some finding"],
        critique={"sufficient": True, "issues": [messy_issue], "recommended_action": "proceed"},
    )

    report = generate_report(state)["final_report"]

    assert "\n  \"sufficient\"" not in report  # raw JSON fragment not reproduced
    assert "...(truncated)" in report
    # The bullet is present but bounded in length.
    limitations_section = report.split("## Limitations")[1].split("## Sources")[0]
    for line in limitations_section.strip().splitlines():
        if line.strip():
            assert len(line) <= 220 + len("...(truncated)") + len("- Quality issue noted during research: ")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
