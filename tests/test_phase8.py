"""
Tests for Phase 8 (guardrails + failure handling).

Scope of this file is deliberately narrow: it covers the two guardrails
introduced in Phase 8 (app/guardrails.py) - input task validation and
page_reader URL validation - plus main.py's empty-report safety net.

Everything else Phase 8 asks for (API failure, search failure, timeouts,
malformed LLM JSON, max iterations, missing sources, unsupported claims,
empty report from the reporter itself) already has dedicated coverage in
tests/test_phase2.py through tests/test_phase7.py and is not repeated
here - see app/guardrails.py's module docstring for exactly where each
lives.

Same philosophy as earlier phases: no real network access, no LLM API
key required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.guardrails import GuardrailError, validate_task, validate_fetch_url
from app.tools.page_reader import read_page


# ---------------------------------------------------------------------------
# validate_task (input guardrail)
# ---------------------------------------------------------------------------

def test_validate_task_strips_and_accepts_normal_question():
    assert validate_task("  What is the price of widget X?  ") == "What is the price of widget X?"


def test_validate_task_rejects_empty():
    with pytest.raises(GuardrailError):
        validate_task("")


def test_validate_task_rejects_whitespace_only():
    with pytest.raises(GuardrailError):
        validate_task("   \n\t  ")


def test_validate_task_rejects_too_long():
    with pytest.raises(GuardrailError):
        validate_task("x" * 3000)


def test_validate_task_rejects_no_word_characters():
    with pytest.raises(GuardrailError):
        validate_task("??? !!! ...")


def test_validate_task_accepts_short_but_real_question():
    assert validate_task("AI?") == "AI?"


# ---------------------------------------------------------------------------
# validate_fetch_url (tool-argument / SSRF guardrail)
# ---------------------------------------------------------------------------

def test_validate_fetch_url_accepts_plain_https():
    assert validate_fetch_url("https://example.com/pricing") == "https://example.com/pricing"


def test_validate_fetch_url_rejects_file_scheme():
    with pytest.raises(GuardrailError):
        validate_fetch_url("file:///etc/passwd")


def test_validate_fetch_url_rejects_ftp_scheme():
    with pytest.raises(GuardrailError):
        validate_fetch_url("ftp://example.com/file")


def test_validate_fetch_url_rejects_missing_host():
    with pytest.raises(GuardrailError):
        validate_fetch_url("https:///no-host-here")


def test_validate_fetch_url_rejects_localhost_hostname():
    with pytest.raises(GuardrailError):
        validate_fetch_url("http://localhost:8080/admin")


def test_validate_fetch_url_rejects_loopback_ip():
    with pytest.raises(GuardrailError):
        validate_fetch_url("http://127.0.0.1/secret")


def test_validate_fetch_url_rejects_private_ip():
    with pytest.raises(GuardrailError):
        validate_fetch_url("http://10.0.0.5/internal")


def test_validate_fetch_url_rejects_link_local_cloud_metadata_ip():
    # 169.254.169.254 is the AWS/GCP/Azure instance-metadata address - a
    # classic SSRF target.
    with pytest.raises(GuardrailError):
        validate_fetch_url("http://169.254.169.254/latest/meta-data/")


# ---------------------------------------------------------------------------
# read_page wires validate_fetch_url in (no real HTTP call should happen
# for a blocked URL - requests.get is monkeypatched to blow up if reached)
# ---------------------------------------------------------------------------

def test_read_page_refuses_blocked_url_without_making_a_request(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        raise AssertionError("requests.get should not be called for a blocked URL")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    with pytest.raises(RuntimeError):
        read_page("http://127.0.0.1/secret")


def test_read_page_refuses_non_http_scheme_without_making_a_request(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        raise AssertionError("requests.get should not be called for a blocked scheme")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    with pytest.raises(RuntimeError):
        read_page("file:///etc/passwd")
