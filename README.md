# ResearchPilot (Phase 2 — real agentic loop)

An autonomous web research and report-generation agent. Phase 1 was a
linear **PLAN → ACT → OBSERVE → UPDATE STATE → FINAL** pipeline. Phase 2
replaces the fixed researcher→reporter step with a genuine conditional
loop: an LLM-driven evaluator decides, after each research pass, whether
there's enough evidence to report or whether more research is needed. A
full critic (source-relevance checks, structured issue list), richer
tools, memory, and evaluation are **not** implemented yet — they come in
later phases.

## What it does right now

1. You give it a research question via the CLI.
2. **Planner** (LLM) breaks it into 3–5 concrete research steps.
3. **Researcher** runs a real web search for each *pending* step
   (DuckDuckGo HTML endpoint, no API key needed) and records findings +
   sources.
4. **Evaluator** (LLM) checks whether the findings so far are enough to
   answer the goal. If not, it proposes 1–3 new search queries and the
   plan is extended — routing back to the researcher. If so (or the
   iteration cap is hit), it moves on.
5. **Reporter** (LLM) synthesizes the findings into a final answer, with
   sources listed, and saves it to `output/`.

Orchestration is a LangGraph graph with a conditional loop:

```
                         ┌────────────────────────┐
                         │                         │
                         ▼                         │
START -> planner -> researcher -> evaluator ───────┘  (evidence insufficient,
                                       │                under iteration cap)
                                       ▼
                                   reporter -> END      (evidence sufficient,
                                                          or cap hit)
```

The evaluator — not a fixed sequence — decides which branch to take each
time, based on current state (findings so far, what's already been
searched, iteration count). See `app/evaluator.py`.

## Project structure

```
researchpilot/
├── app/
│   ├── main.py          # CLI entrypoint
│   ├── state.py         # shared Pydantic AgentState
│   ├── llm_provider.py  # configurable LLM wrapper (Anthropic/Groq)
│   ├── graph.py          # LangGraph wiring, incl. conditional loop
│   ├── planner.py
│   ├── researcher.py     # now dedups against completed_steps
│   ├── evaluator.py      # NEW (Phase 2): evidence check + routing decision
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
| `RESEARCHPILOT_MAX_ITERATIONS` | Max research/evaluate loop iterations before forcing a report (default `3`). Prevents infinite loops. |

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

`tests/test_mvp.py` (Phase 1) and `tests/test_phase2.py` (Phase 2) cover
`AgentState`, the calculator tool, JSON-extraction (planner + evaluator),
the evaluator's routing decision and iteration cap, and the researcher's
dedup-against-`completed_steps` logic — all without needing network access
or an API key (network calls are monkeypatched out in the researcher
tests). They do **not** cover live LLM/web-search calls themselves, since
those require real credentials and internet access; verify those manually
with `python -m app.main "..."` (see "How this was tested" below).

## How this was tested

In the sandboxed environment used to build this phase:
- Every file (Phase 1 + Phase 2) compiles cleanly (`python -m py_compile`).
- The `pydantic`/`langgraph`/`pytest` packages are **not installed** in
  this sandbox, and it has no network access to install them (`pip
  install` fails with "No matching distribution found" — not a code
  issue, an environment one). So the test suite and
  `app.graph.build_graph()` could not actually be *executed* here; the
  Phase 2 additions were verified by careful code review and syntax
  checking instead.
- Live web search and live LLM calls: **could not be executed** here
  either, for the same reason noted in the Phase 1 section below (no
  general internet egress, no API key configured in this sandbox).

**You should run `pip install -r requirements.txt pytest`, then
`python -m pytest tests/ -v`, then a real `python -m app.main "..."` query
on your own machine** before treating Phase 2 as fully validated
end-to-end. A good test query is one with an intentionally incomplete
first search topic, so you can watch the evaluator trigger a second
iteration (e.g. a "current pricing" question, where the first search may
miss the official source).

## Known limitations (Phase 2 stage, expected)

- The evaluator is a lightweight sufficiency check, not the full Phase 4
  critic — it doesn't verify that individual claims are source-supported,
  check source relevance, or emit the structured
  `sufficient/issues/recommended_action` schema. That's Phase 4.
- If the evaluator's LLM call fails (bad JSON, API error), the code
  fails *open* — it assumes evidence is sufficient and moves to the
  reporter, rather than looping forever or crashing. This is a deliberate
  simplicity/robustness tradeoff for this phase.
- Web search still uses a no-key DuckDuckGo HTML scrape — fine for this
  stage, but more fragile than a paid search API.
- The calculator tool exists and is unit-tested but is not yet called by
  the agent loop (that wiring comes in Phase 3, when tool selection is
  added).
- Report formatting is plain text + a source list, not the full
  multi-section report format (Phase 5).
- No persistent memory across runs yet (Phase 6).
