"""Deterministic Phase 7 evaluation for completed ResearchPilot runs.

This module evaluates observable run artifacts only: the original task,
final report, findings, sources, and iteration count. It intentionally does
not inspect hidden reasoning and does not call an LLM by default.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from app.state import AgentState

REQUIRED_SECTIONS = (
    "Executive Summary",
    "Research Question",
    "Methodology",
    "Key Findings",
    "Comparison / Analysis",
    "Limitations",
    "Sources",
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+.#/-]*", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
_HEADING_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+(.+?)\s*$")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "what", "which", "with", "vs", "versus", "compare",
    "research", "report", "find", "key",
}


class EvaluationResult(BaseModel):
    """Structured score plus transparent deterministic diagnostics."""

    relevance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    source_coverage: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    format_quality: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    checks: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in _STOPWORDS
    }


def _extract_sections(report: str) -> dict[str, str]:
    matches = list(_HEADING_RE.finditer(report or ""))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        sections[match.group(1).strip()] = report[start:end].strip()
    return sections


def check_required_sections(report: str) -> dict[str, bool]:
    sections = _extract_sections(report)
    return {name: bool(sections.get(name, "").strip()) for name in REQUIRED_SECTIONS}


def _normalized_url(url: str) -> str:
    return (url or "").strip().rstrip("/).,;\"'")


def source_coverage_check(state: AgentState) -> tuple[float, dict[str, Any], list[str]]:
    """Check whether report citations/URLs correspond to collected sources."""
    collected = {_normalized_url(s.url) for s in state.sources if s.url}
    cited = {_normalized_url(u) for u in _URL_RE.findall(state.final_report or "")}
    matched = cited & collected
    unknown = cited - collected
    missing_from_report = collected - cited

    warnings: list[str] = []
    if unknown:
        warnings.append("Report contains URL(s) not present in collected sources.")

    if not collected:
        score = 0.0
    else:
        recall = len(matched) / len(collected)
        precision = len(matched) / len(cited) if cited else 0.0
        score = (recall + precision) / 2.0

    return round(score, 4), {
        "collected_urls": sorted(collected),
        "cited_urls": sorted(cited),
        "matched_urls": sorted(matched),
        "unknown_cited_urls": sorted(unknown),
        "uncited_collected_urls": sorted(missing_from_report),
    }, warnings


def _support_corpus(state: AgentState) -> set[str]:
    chunks = list(state.findings)
    chunks.extend(f"{s.title} {s.snippet} {s.url}" for s in state.sources)
    return _tokens("\n".join(chunks))


def grounding_check(state: AgentState) -> tuple[float, dict[str, Any], list[str]]:
    """Flag key-finding bullets with weak lexical support in gathered evidence.

    This is deliberately conservative and deterministic. It is not semantic
    entailment and must not be interpreted as proof that a claim is true.
    """
    sections = _extract_sections(state.final_report)
    key_findings = sections.get("Key Findings", "")
    bullets = _BULLET_RE.findall(key_findings)
    if not bullets and key_findings.strip() and "(no findings gathered)" not in key_findings.lower():
        bullets = [line.strip() for line in key_findings.splitlines() if line.strip()]

    corpus = _support_corpus(state)
    unsupported: list[str] = []
    supported = 0

    for claim in bullets:
        claim_tokens = _tokens(claim)
        if not claim_tokens:
            continue
        overlap = len(claim_tokens & corpus) / len(claim_tokens)
        # 50% lexical overlap is a transparent heuristic, not factual verification.
        if overlap >= 0.5:
            supported += 1
        else:
            unsupported.append(claim)

    considered = supported + len(unsupported)
    if considered == 0:
        score = 0.0 if not state.findings else 0.5
    else:
        score = supported / considered

    warnings = []
    if unsupported:
        warnings.append(f"{len(unsupported)} key-finding claim(s) had weak lexical support in gathered evidence.")

    return round(score, 4), {
        "claims_considered": considered,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "method": "key-finding lexical overlap >= 0.50 against findings/source metadata",
    }, warnings


def relevance_check(state: AgentState) -> tuple[float, dict[str, Any]]:
    """Measure whether important task terms are reflected in the report."""
    goal_tokens = _tokens(state.user_goal)
    report_tokens = _tokens(state.final_report)
    if not goal_tokens:
        score = 0.0
    else:
        score = len(goal_tokens & report_tokens) / len(goal_tokens)
    return round(score, 4), {
        "goal_terms": sorted(goal_tokens),
        "matched_goal_terms": sorted(goal_tokens & report_tokens),
    }


def format_quality_check(report: str) -> tuple[float, dict[str, Any]]:
    checks = check_required_sections(report)
    score = sum(checks.values()) / len(REQUIRED_SECTIONS)
    return round(score, 4), {"required_sections": checks}


def completeness_check(state: AgentState, allowed_iterations: int) -> tuple[float, dict[str, Any], list[str]]:
    """Observable completion checks; no subjective answer-quality judgment."""
    sections = _extract_sections(state.final_report)
    has_findings = bool(state.findings) and bool(sections.get("Key Findings", "").strip())
    has_sources = bool(state.sources)
    has_summary = bool(sections.get("Executive Summary", "").strip())
    has_limitations = bool(sections.get("Limitations", "").strip())
    within_limit = state.iteration <= allowed_iterations
    no_unresolved_gaps = not bool(state.missing_information)

    checks = {
        "findings_present": has_findings,
        "sources_present": has_sources,
        "executive_summary_present": has_summary,
        "limitations_present": has_limitations,
        "no_unresolved_missing_information": no_unresolved_gaps,
        "terminated_within_iteration_limit": within_limit,
    }
    score = sum(checks.values()) / len(checks)
    warnings = []
    if not has_findings:
        warnings.append("Research run has no usable findings.")
    if not has_sources:
        warnings.append("Research run has no collected sources.")
    if not no_unresolved_gaps:
        warnings.append(
            f"Run ended with {len(state.missing_information)} unresolved information gap(s)."
        )
    if not within_limit:
        warnings.append(
            f"Run iteration count ({state.iteration}) exceeded allowed limit ({allowed_iterations})."
        )
    return round(score, 4), checks, warnings


def evaluate_run(state: AgentState, allowed_iterations: int = 3) -> EvaluationResult:
    """Evaluate one completed run using deterministic, observable checks."""
    relevance, relevance_details = relevance_check(state)
    completeness, completeness_details, completeness_warnings = completeness_check(
        state, allowed_iterations
    )
    source_coverage, source_details, source_warnings = source_coverage_check(state)
    grounding, grounding_details, grounding_warnings = grounding_check(state)
    format_quality, format_details = format_quality_check(state.final_report)

    component_scores = [relevance, completeness, source_coverage, grounding, format_quality]
    overall = round(sum(component_scores) / len(component_scores), 4)

    return EvaluationResult(
        relevance=relevance,
        completeness=completeness,
        source_coverage=source_coverage,
        grounding=grounding,
        format_quality=format_quality,
        overall=overall,
        checks={
            "relevance": relevance_details,
            "completeness": completeness_details,
            "source_coverage": source_details,
            "grounding": grounding_details,
            "format_quality": format_details,
        },
        warnings=completeness_warnings + source_warnings + grounding_warnings,
    )
