"""
Tests for Phase 3 (page reader + report writer tools, and tool selection).

Same philosophy as test_mvp.py / test_phase2.py: no real network access
and no LLM API key required.
- `choose_tool` is pure logic, tested directly.
- `read_page` is tested against a canned HTML response via monkeypatching
  `requests.get`, so no real HTTP call is made.
- `write_report` is tested against a real temp directory (it's a pure
  filesystem operation, safe to exercise for real).
- The researcher's tool routing is tested by monkeypatching all three
  underlying tool functions, so we can verify each step type reaches the
  tool it should without doing real network/file I/O.

Full live behaviour (a real run that actually reads a page or does a
calculation the evaluator asked for) is verified manually via
`python -m app.main "..."`, as with earlier phases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tool_selector import choose_tool, WEB_SEARCH, PAGE_READER, CALCULATOR
from app.tools.page_reader import read_page
from app.tools.report_writer import write_report
from app.state import AgentState, Source
import app.researcher as researcher_module
import app.evaluator as evaluator_module
from app.evaluator import _format_known_sources, _drop_hallucinated_urls, evaluate_evidence


# ---------------------------------------------------------------------------
# tool_selector
# ---------------------------------------------------------------------------

def test_choose_tool_plain_question_is_web_search():
    tool, tool_input = choose_tool("Search for PostgreSQL vs MySQL write performance")
    assert tool == WEB_SEARCH
    assert tool_input == "Search for PostgreSQL vs MySQL write performance"


def test_choose_tool_detects_url_as_page_reader():
    step = "Read more detail at https://example.com/pricing-page"
    tool, tool_input = choose_tool(step)
    assert tool == PAGE_READER
    assert tool_input == "https://example.com/pricing-page"


def test_choose_tool_strips_trailing_punctuation_from_url():
    tool, tool_input = choose_tool("See https://example.com/docs).")
    assert tool == PAGE_READER
    assert tool_input == "https://example.com/docs"


def test_choose_tool_calculate_prefix_is_calculator():
    tool, tool_input = choose_tool("Calculate: 49.99 * 12")
    assert tool == CALCULATOR
    assert tool_input == "49.99 * 12"


def test_choose_tool_bare_expression_is_calculator():
    tool, tool_input = choose_tool("120 - 99.99")
    assert tool == CALCULATOR
    assert tool_input == "120 - 99.99"


def test_choose_tool_does_not_misclassify_a_question_with_a_number():
    # Contains a digit, but is not a bare arithmetic expression -> web_search.
    tool, tool_input = choose_tool("Search for iPhone 15 battery life")
    assert tool == WEB_SEARCH


# ---------------------------------------------------------------------------
# page_reader
# ---------------------------------------------------------------------------

_SAMPLE_HTML = """
<html>
  <head><title>Example Pricing</title></head>
  <body>
    <nav>Home | Pricing | About</nav>
    <script>console.log('ignore me')</script>
    <p>Our plan costs $49.99 per month.</p>
    <p>Annual billing saves 20%.</p>
    <footer>Copyright 2026</footer>
  </body>
</html>
"""


class _FakeResponse:
    def __init__(self, text: str = "", status: int = 200, content_type: str = "text/html", content: bytes = b""):
        self.text = text
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        # Real requests.Response.content is the raw bytes; default to the
        # text encoded, which is fine for the HTML-only tests below.
        self.content = content or text.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")


def test_read_page_extracts_title_and_paragraphs(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(_SAMPLE_HTML)

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    page = read_page("https://example.com/pricing")
    assert page.title == "Example Pricing"
    assert "$49.99 per month" in page.text
    assert "Annual billing saves 20%" in page.text
    # nav/script/footer text must not leak into extracted content.
    assert "Home | Pricing" not in page.text
    assert "console.log" not in page.text
    assert "Copyright" not in page.text


def test_read_page_raises_on_network_failure(monkeypatch):
    import pytest
    import requests

    def fake_get(url, headers=None, timeout=None):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    with pytest.raises(RuntimeError):
        read_page("https://example.com/unreachable")


def test_read_page_raises_when_no_text_found(monkeypatch):
    import pytest

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse("<html><body><nav>only nav</nav></body></html>")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    with pytest.raises(RuntimeError):
        read_page("https://example.com/empty")


def test_read_page_truncates_long_text(monkeypatch):
    long_paragraph = "word " * 2000  # far more than max_chars
    html = f"<html><head><title>T</title></head><body><p>{long_paragraph}</p></body></html>"

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(html)

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    page = read_page("https://example.com/long", max_chars=500)
    assert len(page.text) <= 504  # 500 + "..." plus a little slack
    assert page.text.endswith("...")


# ---------------------------------------------------------------------------
# page_reader: PDF support
# (regression test for the live run where a real Anthropic pricing PDF came
#  back "found no readable text" because it was parsed as HTML)
#
# Rather than authoring real PDF bytes (which would need an extra
# dependency like reportlab just to build a test fixture), these tests
# fake `PdfReader` itself - same monkeypatch-the-boundary approach used
# for `requests.get` elsewhere in this file.
# ---------------------------------------------------------------------------

class _FakePdfPage:
    def __init__(self, text: str = "", raise_on_extract: bool = False):
        self._text = text
        self._raise_on_extract = raise_on_extract

    def extract_text(self):
        if self._raise_on_extract:
            raise ValueError("corrupt page content stream")
        return self._text


class _FakePdfReader:
    """Stand-in for pypdf.PdfReader, constructed the same way (from a stream)."""

    def __init__(self, stream, *, pages=None, title=None, raise_on_init=False):
        if raise_on_init:
            from pypdf.errors import PdfReadError
            raise PdfReadError("could not parse PDF")
        self.pages = pages or []
        self.metadata = type("Meta", (), {"title": title})() if title else None


def _patch_pdf_reader(monkeypatch, **kwargs):
    """Patch app.tools.page_reader.PdfReader to always return a _FakePdfReader(**kwargs)."""
    def factory(stream):
        return _FakePdfReader(stream, **kwargs)
    monkeypatch.setattr("app.tools.page_reader.PdfReader", factory)


def test_read_page_extracts_text_from_pdf_by_content_type(monkeypatch):
    _patch_pdf_reader(
        monkeypatch,
        pages=[_FakePdfPage("The official price is 49.99 per month.")],
        title="Pricing Sheet",
    )

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(content_type="application/pdf", content=b"%PDF-fake-bytes")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    page = read_page("https://example.com/pricing-sheet")
    assert "49.99" in page.text
    assert page.title == "Pricing Sheet"


def test_read_page_detects_pdf_by_url_suffix_when_no_content_type(monkeypatch):
    _patch_pdf_reader(monkeypatch, pages=[_FakePdfPage("Annual plan costs 599.88.")])

    def fake_get(url, headers=None, timeout=None):
        # No/blank Content-Type header from the server - must still detect
        # this as a PDF from the ".pdf" URL suffix, not fall through to the
        # HTML branch (which would find no <p> tags in raw PDF bytes).
        return _FakeResponse(content_type="", content=b"%PDF-fake-bytes")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    page = read_page("https://example.com/files/pricing.pdf")
    assert "599.88" in page.text


def test_read_page_combines_text_across_multiple_pdf_pages(monkeypatch):
    _patch_pdf_reader(
        monkeypatch,
        pages=[_FakePdfPage("Page one: 49.99/mo."), _FakePdfPage("Page two: 599.88/yr.")],
    )

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(content_type="application/pdf", content=b"%PDF-fake-bytes")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    page = read_page("https://example.com/multi-page.pdf")
    assert "49.99/mo" in page.text
    assert "599.88/yr" in page.text


def test_read_page_skips_unparsable_pdf_page_without_failing_whole_doc(monkeypatch):
    _patch_pdf_reader(
        monkeypatch,
        pages=[_FakePdfPage(raise_on_extract=True), _FakePdfPage("Good page: 49.99.")],
    )

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(content_type="application/pdf", content=b"%PDF-fake-bytes")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    page = read_page("https://example.com/partially-corrupt.pdf")
    assert "49.99" in page.text


def test_read_page_raises_on_unparsable_pdf(monkeypatch):
    import pytest

    _patch_pdf_reader(monkeypatch, raise_on_init=True)

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(content_type="application/pdf", content=b"not a real pdf")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    with pytest.raises(RuntimeError):
        read_page("https://example.com/broken.pdf")


def test_read_page_raises_on_pdf_with_no_extractable_text(monkeypatch):
    import pytest

    _patch_pdf_reader(monkeypatch, pages=[_FakePdfPage("")])  # e.g. a scanned/image-only PDF

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(content_type="application/pdf", content=b"%PDF-fake-bytes")

    monkeypatch.setattr("app.tools.page_reader.requests.get", fake_get)

    with pytest.raises(RuntimeError):
        read_page("https://example.com/scanned.pdf")


# ---------------------------------------------------------------------------
# report_writer
# ---------------------------------------------------------------------------

def test_write_report_creates_file_with_expected_content(tmp_path):
    path = write_report("What is X?", "X is Y.\n\n---\nSources:\n- (no sources)", output_dir=tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith("report_") and path.name.endswith(".md")

    content = path.read_text(encoding="utf-8")
    assert "# Research Report" in content
    assert "**Goal:** What is X?" in content
    assert "X is Y." in content


def test_write_report_creates_missing_output_dir(tmp_path):
    nested = tmp_path / "nested" / "output"
    path = write_report("goal", "body", output_dir=nested)
    assert path.exists()
    assert nested.exists()


# ---------------------------------------------------------------------------
# researcher tool routing
# ---------------------------------------------------------------------------

def test_researcher_routes_url_step_to_page_reader(monkeypatch):
    from app.tools.page_reader import PageContent

    calls = []

    def fake_read_page(url, timeout=10, max_chars=4000):
        calls.append(url)
        return PageContent(url=url, title="T", text="some page text")

    def fake_web_search(query, max_results=3):
        raise AssertionError("web_search should not be called for a URL step")

    monkeypatch.setattr(researcher_module, "read_page", fake_read_page)
    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    state = AgentState(user_goal="g", plan=["Read https://example.com/pricing for details"])
    update = researcher_module.research(state)

    assert calls == ["https://example.com/pricing"]
    assert update["tool_history"][0].tool == "page_reader"
    assert any(isinstance(s, Source) and s.url == "https://example.com/pricing" for s in update["sources"])


def test_researcher_routes_calculation_step_to_calculator(monkeypatch):
    def fake_web_search(query, max_results=3):
        raise AssertionError("web_search should not be called for a calculation step")

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    state = AgentState(user_goal="g", plan=["Calculate: 100 - 25"])
    update = researcher_module.research(state)

    assert update["tool_history"][0].tool == "calculator"
    assert "result = 75" in update["findings"][0]


def test_researcher_still_routes_plain_step_to_web_search(monkeypatch):
    from app.tools.web_search import SearchResult

    def fake_web_search(query, max_results=3):
        return [SearchResult(title="T", url="http://x.com", snippet="s")]

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    state = AgentState(user_goal="g", plan=["Search for background information"])
    update = researcher_module.research(state)

    assert update["tool_history"][0].tool == "web_search"


# ---------------------------------------------------------------------------
# evaluator: known-sources grounding for proposed URLs
# (regression test for the live run where the evaluator invented a plausible
#  but non-existent PDF URL instead of copying the real one already in
#  state.sources, and page_reader 404'd on it)
# ---------------------------------------------------------------------------

def test_format_known_sources_dedupes_by_url():
    state = AgentState(
        user_goal="g",
        sources=[
            Source(url="http://a.com", title="A"),
            Source(url="http://a.com", title="A duplicate"),
            Source(url="http://b.com", title="B"),
        ],
    )
    text = _format_known_sources(state)
    assert text.count("http://a.com") == 1
    assert "http://b.com" in text


def test_format_known_sources_handles_no_sources():
    state = AgentState(user_goal="g")
    assert _format_known_sources(state) == "(no sources yet)"


def test_drop_hallucinated_urls_removes_unknown_url():
    known = {"http://real.com/page"}
    queries = [
        "http://real.com/page",
        "http://fabricated.com/guessed.pdf",
        "a plain search query",
    ]
    kept = _drop_hallucinated_urls(queries, known)
    assert kept == ["http://real.com/page", "a plain search query"]


def test_drop_hallucinated_urls_strips_trailing_punctuation_before_checking():
    known = {"http://real.com/page"}
    kept = _drop_hallucinated_urls(["See http://real.com/page."], known)
    assert kept == ["See http://real.com/page."]


def test_evaluate_evidence_drops_fabricated_url_from_llm(monkeypatch):
    import json as json_module

    class FakeLLM:
        def complete(self, prompt, system="", max_tokens=700):
            # Sanity-check the prompt actually includes the known source,
            # so the LLM has a real URL available to copy.
            assert "http://real.com/pricing" in prompt
            return json_module.dumps({
                "sufficient": False,
                "missing_information": ["exact pricing"],
                "additional_queries": [
                    "http://fabricated.com/guessed-pricing.pdf",  # not a real source -> must be dropped
                    "official pricing search query",
                ],
            })

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: FakeLLM())

    state = AgentState(
        user_goal="What does X cost?",
        findings=["Step 'search': some snippet"],
        sources=[Source(url="http://real.com/pricing", title="Real pricing page")],
    )
    update = evaluate_evidence(state)

    new_plan = update["plan"]
    assert "http://fabricated.com/guessed-pricing.pdf" not in new_plan
    assert "official pricing search query" in new_plan


def test_evaluate_evidence_keeps_url_that_matches_known_source(monkeypatch):
    import json as json_module

    class FakeLLM:
        def complete(self, prompt, system="", max_tokens=700):
            return json_module.dumps({
                "sufficient": False,
                "missing_information": ["deeper detail"],
                "additional_queries": ["http://real.com/pricing"],
            })

    monkeypatch.setattr(evaluator_module, "get_llm_client", lambda: FakeLLM())

    state = AgentState(
        user_goal="What does X cost?",
        findings=["Step 'search': some snippet"],
        sources=[Source(url="http://real.com/pricing", title="Real pricing page")],
    )
    update = evaluate_evidence(state)

    assert "http://real.com/pricing" in update["plan"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
