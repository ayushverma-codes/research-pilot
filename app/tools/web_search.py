"""
Web search tool.

Uses DuckDuckGo's HTML endpoint (no API key required) so the MVP works out
of the box. This is a known limitation: no-key scraping is less reliable
than a paid search API and may break if DuckDuckGo changes its markup.
Swapping in a proper search API later only requires changing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"
HEADERS = {"User-Agent": "Mozilla/5.0 (ResearchPilot Agent)"}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5, timeout: int = 10) -> List[SearchResult]:
    """Run a web search and return a list of SearchResult. Raises on network failure."""
    try:
        resp = requests.post(
            SEARCH_URL,
            data={"q": query},
            headers=HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"web_search failed for query '{query}': {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[SearchResult] = []

    for result_div in soup.select("div.result")[:max_results]:
        title_tag = result_div.select_one("a.result__a")
        snippet_tag = result_div.select_one("a.result__snippet, div.result__snippet")
        if not title_tag:
            continue
        results.append(
            SearchResult(
                title=title_tag.get_text(strip=True),
                url=title_tag.get("href", ""),
                snippet=snippet_tag.get_text(strip=True) if snippet_tag else "",
            )
        )

    return results
