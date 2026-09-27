"""
Guardrails (Phase 8).

Practical, mostly deterministic safety checks around the two places things
can go wrong that are NOT already handled elsewhere in the pipeline:

  - INPUT     : the raw task text from the user, before a single LLM/tool
                call is made (empty, malformed, or clearly unsupported).
  - TOOL ARGS : a URL proposed for page_reader to fetch. This is the one
                tool argument in the whole system that is attacker/LLM
                influenced *and* reaches outside the process (calculator
                only evaluates arithmetic via `ast`; web_search only takes
                a query string) - so it's the one real SSRF surface.

Everything else Phase 8 asks for is already handled at the point it
happens, and is intentionally not duplicated here:
  - API/search failure, timeouts           -> app/llm_provider.py (LLMError,
                                               _run_with_hard_timeout),
                                               app/researcher.py (per-tool
                                               try/except)
  - malformed LLM JSON                     -> app/planner.py, app/evaluator.py
                                               (_extract_json_array/_object)
  - invalid calculator expressions         -> app/tools/calculator.py
                                               (ast-based, raises CalculatorError)
  - max iterations                         -> app/evaluator.py (MAX_ITERATIONS)
  - missing sources / unsupported claims /
    empty report                          -> app/reporter.py (grounding gate,
                                               deterministic fallback narrative,
                                               "(no sources)" / "(no findings
                                               gathered)" placeholders)
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

MIN_TASK_CHARS = 3
MAX_TASK_CHARS = 2000

ALLOWED_URL_SCHEMES = {"http", "https"}

# Hostnames that are local by convention rather than by IP literal. Kept as
# an exact-match set (no DNS resolution here - see _is_blocked_host's
# docstring for why) so this stays a pure, offline, deterministic check
# like the rest of the tool layer.
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",  # common cloud metadata endpoint hostname
}


class GuardrailError(ValueError):
    """Raised when input or a tool argument fails a guardrail check.

    The message is written to be shown directly to the user/caller.
    """


def validate_task(user_goal: str) -> str:
    """
    Validate a raw user task before any planning/research begins.

    Returns the cleaned task string on success, or raises GuardrailError
    for:
      - empty/whitespace-only input
      - input so long it would blow up planner/critic/report prompt
        budgets for no real benefit (malformed / abusive input)
      - input with no actual word characters at all (not a task in any
        recognizable sense - e.g. pure punctuation or emoji)
    """
    cleaned = (user_goal or "").strip()

    if not cleaned:
        raise GuardrailError("research question cannot be empty.")

    if len(cleaned) > MAX_TASK_CHARS:
        raise GuardrailError(
            f"research question is too long ({len(cleaned)} chars; "
            f"max {MAX_TASK_CHARS}). Please shorten it."
        )

    if len(cleaned) < MIN_TASK_CHARS or not re.search(r"[A-Za-z0-9]", cleaned):
        raise GuardrailError(
            "research question doesn't look like a real question - "
            "please describe what you'd like researched."
        )

    return cleaned


def _is_blocked_host(hostname: str) -> bool:
    """
    True if hostname is a known-local name, or a literal IP that is
    loopback/private/link-local/reserved/multicast.

    Deliberately does NOT resolve hostnames via DNS: this keeps the check
    pure and offline (consistent with the rest of the tool layer, which is
    tested against canned responses with no real network access - see
    tests/test_phase3.py), and avoids the check itself making a network
    call before requests.get does. It catches the common, cheap cases (an
    LLM- or search-result-supplied URL that is a raw internal/loopback IP,
    or an obviously-local hostname) without pretending to be a complete
    defense against DNS rebinding.
    """
    host = hostname.lower().strip("[]")  # strip IPv6 literal brackets, if any

    if host in _BLOCKED_HOSTNAMES:
        return True

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False

    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast


def validate_fetch_url(url: str) -> str:
    """
    Validate a URL before page_reader fetches it.

    Raises GuardrailError for anything that isn't a plain public http(s)
    URL: a disallowed scheme (e.g. file://, ftp://), a missing host, or a
    host that is (or resolves to) a loopback/private/link-local address.
    Returns the (unchanged) URL on success.
    """
    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise GuardrailError(f"unsupported URL scheme in '{url}' (only http/https allowed).")

    if not parsed.hostname:
        raise GuardrailError(f"could not determine a host in '{url}'.")

    if _is_blocked_host(parsed.hostname):
        raise GuardrailError(f"refusing to fetch internal/private address: '{url}'.")

    return url
