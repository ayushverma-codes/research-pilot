"""Phase 7 deterministic evaluation tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import (
    EvaluationResult,
    check_required_sections,
    evaluate_run,
    grounding_check,
    source_coverage_check,
)
from app.state import AgentState, Source


def _report(key_findings="- Alpha costs $10 per month.", sources="- Alpha (https://alpha.example/pricing)"):
    return f"""# Research Report

## Executive Summary
Alpha pricing was researched.

## Research Question
What does Alpha cost?

## Methodology
One source was reviewed.

## Key Findings
{key_findings}

## Comparison / Analysis
The available evidence answers the pricing question.

## Limitations
- Only one source was gathered.

## Sources
{sources}
"""


def test_evaluation_result_creation():
    result = EvaluationResult(
        relevance=1,
        completeness=1,
        source_coverage=1,
        grounding=1,
        format_quality=1,
        overall=1,
    )
    assert result.overall == 1
    assert result.warnings == []


def test_section_checks_detect_missing_section():
    report = _report().replace("## Limitations\n- Only one source was gathered.\n\n", "")
    checks = check_required_sections(report)
    assert checks["Executive Summary"] is True
    assert checks["Limitations"] is False


def test_source_coverage_matches_collected_urls_and_flags_unknown():
    state = AgentState(
        user_goal="What does Alpha cost?",
        sources=[Source(url="https://alpha.example/pricing", title="Alpha")],
        final_report=_report(sources=(
            "- Alpha (https://alpha.example/pricing)\n"
            "- Unknown (https://unknown.example/source)"
        )),
    )
    score, details, warnings = source_coverage_check(state)
    assert score == 0.75  # recall=1.0, precision=0.5
    assert details["unknown_cited_urls"] == ["https://unknown.example/source"]
    assert warnings


def test_grounding_flags_unsupported_key_finding():
    state = AgentState(
        user_goal="What does Alpha cost?",
        findings=["Alpha costs $10 per month."],
        sources=[Source(url="https://alpha.example/pricing", title="Alpha", snippet="Alpha costs $10 per month.")],
        final_report=_report(key_findings=(
            "- Alpha costs $10 per month.\n"
            "- Alpha has 99.999 percent uptime and free enterprise support."
        )),
    )
    score, details, warnings = grounding_check(state)
    assert score == 0.5
    assert len(details["unsupported_claims"]) == 1
    assert warnings


def test_incomplete_research_scores_completeness_down():
    state = AgentState(
        user_goal="Compare Alpha and Beta pricing.",
        findings=[],
        sources=[],
        missing_information=["Beta pricing"],
        iteration=2,
        final_report=_report(key_findings="(no findings gathered)", sources="(no sources)"),
    )
    result = evaluate_run(state, allowed_iterations=3)
    assert result.completeness < 1.0
    assert result.source_coverage == 0.0
    assert any("no usable findings" in warning.lower() for warning in result.warnings)


def test_iteration_limit_behavior_is_recorded():
    state = AgentState(
        user_goal="What does Alpha cost?",
        findings=["Alpha costs $10 per month."],
        sources=[Source(url="https://alpha.example/pricing", title="Alpha", snippet="Alpha costs $10 per month.")],
        iteration=4,
        final_report=_report(),
    )
    result = evaluate_run(state, allowed_iterations=3)
    assert result.checks["completeness"]["terminated_within_iteration_limit"] is False
    assert result.completeness == 0.8333
    assert any("exceeded allowed limit" in warning.lower() for warning in result.warnings)
