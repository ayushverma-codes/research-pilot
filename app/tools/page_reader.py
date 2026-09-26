"""
Page reader tool.

Fetches a single URL and extracts its readable text, so the agent can go
deeper on a specific source than a search snippet allows. Used when a
research step points at a particular page rather than asking a general
question.

Handles two content types:
  - HTML (the common case): extracted via BeautifulSoup, paragraph text
    only, noise tags (nav/script/footer/etc.) stripped first.
  - PDF: many official sources (pricing sheets, spec sheets, whitepapers)
    are published as PDFs rather than HTML pages, and feeding PDF bytes to
    an HTML parser silently finds no text at all. This branch extracts
    text per-page via pypdf instead. Detected by Content-Type header,
    falling back to a ".pdf" URL suffix if the server doesn't send one.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

HEADERS = {"User-Agent": "Mozilla/5.0 (ResearchPilot Agent)"}

# Strip these before extracting text - they're never article content.
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "form"]

# How many PDF pages to read before stopping. Pricing/spec PDFs this tool
# is meant for are short; a very long PDF would just waste time and tokens
# well past `max_chars` anyway.
_MAX_PDF_PAGES = 20


@dataclass
class PageContent:
    url: str
    title: str
    text: str


def _is_pdf(url: str, content_type: str) -> bool:
    return "application/pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf")


def _extract_html(resp: requests.Response) -> tuple[str, str]:
    """Return (title, text) extracted from an HTML response."""
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    return title, text


def _extract_pdf(resp: requests.Response, url: str) -> tuple[str, str]:
    """Return (title, text) extracted from a PDF response, page by page."""
    try:
        reader = PdfReader(io.BytesIO(resp.content))
    except (PdfReadError, ValueError) as e:
        raise RuntimeError(f"read_page could not parse PDF at '{url}': {e}") from e

    title = ""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()

    page_texts = []
    for page in reader.pages[:_MAX_PDF_PAGES]:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a single unparsable page shouldn't fail the whole doc
            continue

    text = "\n".join(t.strip() for t in page_texts if t.strip())
    return title, text


def read_page(url: str, timeout: int = 10, max_chars: int = 4000) -> PageContent:
    """
    Fetch `url` and return its title + main text (truncated to `max_chars`).
    Transparently handles both HTML pages and PDF documents. Raises
    RuntimeError on any network failure, bad status, unparsable PDF, or a
    page with no extractable text, so callers can handle it the same way
    they handle a failed web_search.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"read_page failed for '{url}': {e}") from e

    content_type = resp.headers.get("Content-Type", "")
    if _is_pdf(url, content_type):
        title, text = _extract_pdf(resp, url)
    else:
        title, text = _extract_html(resp)

    if not text:
        raise RuntimeError(f"read_page found no readable text at '{url}'")

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    return PageContent(url=url, title=title, text=text)
