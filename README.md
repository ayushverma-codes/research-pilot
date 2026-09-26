# ResearchPilot (Phase 3 — real tools + tool selection)

An autonomous web research and report-generation agent. Phase 1 was a
linear **PLAN → ACT → OBSERVE → UPDATE STATE → FINAL** pipeline. Phase 2
added a genuine conditional loop: an LLM-driven evaluator decides, after
each research pass, whether there's enough evidence to report or whether
more research is needed. Phase 3 adds two more tools (a page reader and a
report-file writer) and real **tool selection**: each plan step is now
routed to the tool that fits its shape, instead of every step always
going through web search. A full critic (source-relevance checks,
structured issue list), memory, and evaluation are **not** implemented
yet — they come in later phases.

## What it does right now

1. You give it a research question via the CLI.
2. **Planner** (LLM) breaks it into 3–5 concrete research steps.
3. **Researcher** executes each *pending* step, routing it to the tool
   that fits it (see "Tools" below), and records findings + sources.
4. **Evaluator** (LLM) checks whether the findings so far are enough to
   answer the goal. If not, it proposes 1–3 next actions — usually a new
   search query, but it can also point at a specific source URL to read
   in depth, or a calculation to run — and the plan is extended, routing
   back to the researcher. If sufficient (or the iteration cap is hit),
   it moves on.
5. **Reporter** (LLM) synthesizes the findings into a final answer, with
   sources listed.
6. The report is saved to `output/` via the **report writer** tool.

## Tools

| Tool | File | Used for |
|---|---|---|
| `web_search` | `app/tools/web_search.py` | The default: a general research question. DuckDuckGo HTML endpoint, no API key needed. |
| `page_reader` | `app/tools/page_reader.py` | A step that names a specific URL — fetches the page and extracts its readable text, going deeper than a search snippet. Handles both HTML pages and PDF documents (many official pricing/spec sources are PDFs), detected by Content-Type header or `.pdf` URL suffix. |
| `calculator` | `app/tools/calculator.py` | A step that is, or asks for, an arithmetic calculation (e.g. `"Calculate: 49.99 * 12"`). Safe `ast`-based evaluation, no `eval`. |
| `report_writer` | `app/tools/report_writer.py` | Saves the finished report to `output/` with a timestamped filename. |

`app/tool_selector.py` decides which of `web_search` / `page_reader` /
`calculator` handles a given step, from the step's shape (does it contain
a URL? is it a bare arithmetic expression or `"Calculate: ..."`?) — see
"Tool selection" below for why this is deterministic rather than another
LLM call.

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

### Tool selection

Inside the researcher node, each pending step is routed to a tool by
`app.tool_selector.choose_tool`:

```
step contains a URL                        -> page_reader
step is/asks for an arithmetic calculation  -> calculator
otherwise (the common case)                 -> web_search
```

This is deliberately simple pattern matching, not a second LLM call per
step. The *decision about what work is needed* (a new search? a specific
source read in depth? a calculation?) is already made by the LLM-driven
evaluator in Phase 2 — it just expresses that decision as plain text
(a search query, a URL, or `"Calculate: <expr>"`). Tool selection only has
to figure out *how to execute* whatever text it's given, which a few
regexes do reliably and for free, keeping the loop explainable and fast.

## Project structure

```
researchpilot/
├── app/
│   ├── main.py           # CLI entrypoint; saves report via report_writer tool
│   ├── state.py          # shared Pydantic AgentState
│   ├── llm_provider.py   # configurable LLM wrapper (Anthropic/Groq)
│   ├── graph.py          # LangGraph wiring, incl. conditional loop
│   ├── planner.py
│   ├── researcher.py     # NEW (Phase 3): routes each step to a tool via tool_selector
│   ├── tool_selector.py  # NEW (Phase 3): decides web_search / page_reader / calculator
│   ├── evaluator.py      # evidence check + routing decision (can now also
│   │                     #   propose a URL to read, or a calculation)
│   ├── reporter.py
│   └── tools/
│       ├── web_search.py
│       ├── page_reader.py    # NEW (Phase 3): fetch + extract text from one URL
│       ├── calculator.py     # now wired into the agent loop (was unused in Phase 2)
│       └── report_writer.py  # NEW (Phase 3): save the final report to output/
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

`tests/test_mvp.py` (Phase 1), `tests/test_phase2.py` (Phase 2), and
`tests/test_phase3.py` (Phase 3, new) cover `AgentState`, all four tools
(`calculator`, `page_reader`, `report_writer`, and `web_search` indirectly
via the researcher), JSON-extraction (planner + evaluator), the
evaluator's routing decision and iteration cap, `tool_selector.choose_tool`
for every step shape, and the researcher's dedup and tool-routing logic —
all without needing network access or an API key (`requests.get` /
`web_search` / `read_page` are monkeypatched out wherever a test would
otherwise need the network; `write_report` is exercised for real against a
`tmp_path`, since it's a pure filesystem op that's safe to run anywhere).
They do **not** cover live LLM calls or live web search themselves, since
those need real credentials and open internet access; verify those
manually with `python -m app.main "..."` (see "How this was tested"
below).

## How this was tested

In the environment used to build this phase, `pip install -r
requirements.txt pytest` succeeded and every check below was actually
*executed* (not just reviewed):
- `python -m py_compile` on every `app/**/*.py` file: clean.
- `python -m pytest tests/ -v`: **44/44 passed**, including all 21 new
  Phase 3 tests (`tool_selector`; `page_reader` against canned HTML and
  against both real and mocked PDF content; `report_writer` against a
  real temp directory; the researcher's routing to each of the three
  tools; and the evaluator's known-sources grounding for proposed URLs).
- A follow-up fix, made after two real runs surfaced issues live: (1) the
  evaluator was proposing plausible-but-fabricated URLs instead of
  copying real ones from `state.sources` — fixed by giving it an exact
  "Known sources" list to copy from, plus a code-level filter that drops
  any proposed URL not in that list; (2) `page_reader` returned "no
  readable text" for an official pricing PDF, because PDF bytes were
  being parsed as HTML — fixed by adding Content-Type/`.pdf`-suffix
  detection and a `pypdf`-based extraction path. Both are covered by new
  tests, and the PDF path was additionally verified against a real,
  freshly-built PDF byte stream (not just a mocked `PdfReader`) to
  confirm the actual parsing logic works, not only the mock.
- A full offline **integration smoke test**: `app.graph.build_graph()`
  invoked end-to-end with the LLM calls and `web_search`/`read_page`
  monkeypatched to return scripted responses (a first pass that's
  incomplete, a plan step that's a calculation, an evaluator-proposed URL
  for the second pass). This confirmed the whole
  planner -> researcher (routing to web_search, then calculator, then
  page_reader) -> evaluator -> researcher -> evaluator -> reporter ->
  write_report path actually wires together and produces a saved report —
  not just that each piece compiles in isolation.
- Live web search and live LLM calls: this sandbox's network egress is
  restricted to a package-registry allowlist (pypi, npm, GitHub, etc.),
  so DuckDuckGo and arbitrary URLs are not reachable from here, and no
  API key is configured. These paths are exercised by the mocked
  integration test above instead.

**You should still run a real `python -m app.main "..."` query on your
own machine** (with `.env` filled in and open internet access) before
treating Phase 3 as fully validated end-to-end — the mocked test proves
the wiring is correct, not that DuckDuckGo's current HTML markup or a
particular live page still parses as expected. A good test query is one
that plausibly needs a calculation and a deeper look at one source, e.g.
"What would a year of the Notion paid plan cost, and how does that
compare to the free plan's limits?", so you can watch [TOOL] web_search,
[TOOL] calculator, and potentially [TOOL] page_reader all fire in one
run.

## Known limitations (Phase 3 stage, expected)

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
- Tool selection (`app/tool_selector.py`) is deterministic pattern
  matching, not an LLM call — see "Tool selection" above for why that's
  the simplest robust option here, not a corner cut.
- `page_reader` does a single unstructured `<p>` extraction for HTML (no
  JS rendering) and a page-by-page text extraction for PDF via `pypdf` —
  fine for typical article/pricing pages and text-based PDFs, but it
  won't get useful text from a heavily JavaScript-rendered page or a
  scanned/image-only PDF with no text layer (no OCR).
- Report formatting is plain text + a source list, not the full
  multi-section report format (Phase 5).
- No persistent memory across runs yet (Phase 6).
