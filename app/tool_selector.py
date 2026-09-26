"""
Tool selection.

Given a single plan step (as text), decide which tool should execute it.
This is the "Tool Selection" node in the target workflow, kept as simple
deterministic pattern matching rather than a second LLM call per step -
the evaluator (Phase 2) already decides *what* work is needed next (e.g.
proposing a specific URL to read, or a calculation to run); this module
just decides *how* to execute a step once it exists, based on its shape:

  - a step containing a URL                       -> page_reader
  - a step that is, or explicitly asks for, a
    simple arithmetic calculation                 -> calculator
  - everything else (the common case)             -> web_search
"""

from __future__ import annotations

import re
from typing import Tuple

URL_RE = re.compile(r"https?://\S+")
CALC_PREFIX_RE = re.compile(r"^\s*calculate\s*:\s*(.+)$", re.IGNORECASE)
# A step that, ignoring whitespace, is nothing but digits/operators - i.e.
# it IS an arithmetic expression rather than a natural-language question.
BARE_EXPRESSION_RE = re.compile(r"^[\s0-9.()+\-*/%]+$")

WEB_SEARCH = "web_search"
PAGE_READER = "page_reader"
CALCULATOR = "calculator"


def choose_tool(step: str) -> Tuple[str, str]:
    """
    Return (tool_name, tool_input) for a plan step. tool_input is the
    exact string that tool needs (the expression for the calculator, the
    URL for the page reader, or the step text unchanged for web_search).
    """
    calc_match = CALC_PREFIX_RE.match(step)
    if calc_match:
        return CALCULATOR, calc_match.group(1).strip()

    if BARE_EXPRESSION_RE.match(step) and any(c.isdigit() for c in step):
        return CALCULATOR, step.strip()

    url_match = URL_RE.search(step)
    if url_match:
        # Strip common trailing punctuation that isn't part of the URL
        # (e.g. a step written as "...see https://x.com/page.").
        return PAGE_READER, url_match.group(0).rstrip(").,;\"'")

    return WEB_SEARCH, step
