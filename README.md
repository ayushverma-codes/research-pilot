# ResearchPilot (MVP — Phase 1)

An autonomous web research and report-generation agent. This is the
minimal, genuinely working slice of the project: a linear
**PLAN → ACT → OBSERVE → UPDATE STATE → FINAL** pipeline. Conditional
re-planning, a critic, richer tools, memory, and evaluation are **not**
implemented yet — they come in later phases.

## What it does right now

1. You give it a research question via the CLI.
2. **Planner** (LLM) breaks it into 3–5 concrete research steps.
3. **Researcher** runs a real web search for each step (DuckDuckGo HTML
   endpoint, no API key needed) and records findings + sources.
4. **Reporter** (LLM) synthesizes the findings into a final answer, with
   sources listed, and saves it to `output/`.

Orchestration is a linear LangGraph graph:

```
START -> planner -> researcher -> reporter -> END
```

## Project structure

```
researchpilot/
├── app/
│   ├── main.py          # CLI entrypoint
│   ├── state.py         # shared Pydantic AgentState
│   ├── llm_provider.py  # configurable LLM wrapper (Anthropic for now)
│   ├── graph.py          # LangGraph wiring
│   ├── planner.py
│   ├── researcher.py
│   ├── reporter.py
│   └── tools/
│       ├── web_search.py
│       └── calculator.py   # available, not yet wired into the agent loop
├── tests/
├── output/                # generated reports land here
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

```bash
cd researchpilot
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Running

```bash
python -m app.main "What are the key differences between PostgreSQL and MySQL for a high-write OLTP workload?"
```

or run it interactively (no args):

```bash
python -m app.main
```

Output is printed to the terminal and also saved as a timestamped Markdown
file under `output/`.

## Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Which LLM backend to use: `groq` (default) or `anthropic`. |
| `LLM_MODEL` | Model name passed to the provider SDK, e.g. `openai/gpt-oss-120b` (Groq) or `claude-sonnet-4-6` (Anthropic). |
| `GROQ_API_KEY` | Your Groq API key. Required when `LLM_PROVIDER=groq`. Never commit this. |
| `GROQ_REQUESTS_PER_MINUTE` | Client-side throttle for Groq calls (default `25`). See "Rate limiting" below. |
| `ANTHROPIC_API_KEY` | Your Anthropic API key. Only required when `LLM_PROVIDER=anthropic`. |

### Rate limiting (Groq)

Two layers, both in `app/llm_provider.py`:

1. **Proactive throttle** — `RateLimiter` enforces a minimum gap between
   calls based on `GROQ_REQUESTS_PER_MINUTE` (default 25/min, kept under
   typical free-tier ~30 RPM limits), so ordinary use shouldn't trip a 429
   at all.
2. **Reactive backoff** — if a `429` (`RateLimitError`) still happens, the
   client reads the server's `retry-after` header, sleeps exactly that
   long, and retries once (up to `max_retries`, default 3) before failing
   with a clear `LLMError`. Non-rate-limit API errors are not retried.

If you're on a paid/higher-throughput Groq tier, raise
`GROQ_REQUESTS_PER_MINUTE` in `.env` accordingly.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

The included tests cover `AgentState`, the calculator tool, and the
planner's JSON-extraction logic — all without needing network access or an
API key. They do **not** cover the LLM/web-search calls themselves, since
those require live credentials and internet access; verify those manually
with `python -m app.main "..."` (see "Known limitations / how this was
tested" below).

## How this was tested

In the sandboxed environment used to build this MVP:
- `pytest` suite (6 tests): **passed**.
- `app.graph.build_graph()` compiles and produces the expected
  `planner -> researcher -> reporter` graph (verified via
  `get_graph().draw_mermaid()`).
- Live web search: **could not be executed** in that sandbox because its
  network egress only allowlists package registries (pypi, npm, github,
  etc.), not general web hosts — a request to DuckDuckGo returned
  `403 host_not_allowed` from the sandbox's own egress proxy, not from
  DuckDuckGo. The code path is otherwise straightforward
  `requests.post` + BeautifulSoup parsing.
- Live LLM calls: **could not be executed** in that sandbox because no
  `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`) was available there. The
  Groq client's misconfiguration path (missing key → clear `LLMError`) and
  its retry-after parsing / rate-limiter logic **were** unit-tested without
  a live key or network call.

**You should verify both of these on your own machine** (normal internet
access + your own API key) before treating the MVP as fully validated
end-to-end. Nothing above is a code bug — it's a sandbox network/credential
limitation, disclosed here rather than hidden.

## Known limitations (MVP stage, expected)

- No conditional re-planning yet — the plan runs once, straight through,
  regardless of how good the findings are (Phase 2).
- No critic / evidence-sufficiency check yet (Phase 4).
- Web search uses a no-key DuckDuckGo HTML scrape — fine for an MVP, but
  more fragile than a paid search API and can be rate-limited or blocked by
  DuckDuckGo itself in some environments.
- The calculator tool exists and is unit-tested but is not yet called by
  the agent loop (that wiring comes in Phase 3, when tool selection is
  added).
- Report formatting is plain text + a source list, not the full
  multi-section report format (Phase 5).
- No persistent memory across runs yet (Phase 6).
