# ResearchPilot (Phase 8 — guardrails + failure handling)

An autonomous web research and report-generation agent. Phase 1 was a
linear **PLAN → ACT → OBSERVE → UPDATE STATE → FINAL** pipeline. Phase 2
added a genuine conditional loop: an LLM-driven evaluator decides, after
each research pass, whether there's enough evidence to report or whether
more research is needed. Phase 3 added two more tools (a page reader and
a report-file writer) and real **tool selection**: each plan step is
routed to the tool that fits its shape, instead of every step always
going through web search. Phase 4 extended the evaluator into a fuller
**critic**: on top of the coverage check, it also flags quality issues
(unsupported claims, irrelevant/off-topic sources) as structured
`issues`/`recommended_action` output, surfaced (but not yet acted on
beyond coverage) each pass. Phase 5 replaced the plain synthesized
answer with a **structured, multi-section report** (Executive Summary,
Research Question, Methodology, Key Findings, Comparison / Analysis,
Limitations, Sources) — see "Report format" below. Phase 6 adds
**lightweight persistent memory**: every run's queries, successes,
failures and useful source domains are saved to a small JSON file, and
the planner retrieves related past runs before building a new plan — see
"Memory" below. Phase 7 added a separate deterministic evaluation module
for completed runs (`app/evaluation.py`, `python -m app.evaluate`) — see
"Evaluation" below. Phase 8 adds explicit **guardrails**: input
validation on the raw task before any LLM/tool call is made, and outbound
URL validation on `page_reader` (blocked schemes, loopback/private/
link-local targets) — see "Guardrails" below. Everything else Phase 8
calls for (API/search failure, timeouts, malformed LLM JSON, max
iterations, missing sources, unsupported claims, empty report) was
already handled at the point it happens in earlier phases; Phase 8 did
not duplicate that, only closed the two gaps that remained.

## What it does right now

1. You give it a research question via the CLI.
2. **Planner** (LLM) retrieves any related past runs from memory (see
   "Memory" below) and breaks the goal into 3–5 concrete research steps,
   using those past runs as hints where relevant.
3. **Researcher** executes each *pending* step, routing it to the tool
   that fits it (see "Tools" below), and records findings + sources.
4. **Evaluator / Critic** (LLM) checks whether the findings so far are
   enough to answer the goal, *and* flags quality issues (unsupported
   claims, off-topic sources) as structured `issues`. If coverage is
   insufficient, it proposes 1–3 next actions — usually a new search
   query, but it can also point at a specific source URL to read in
   depth, or a calculation to run — and the plan is extended, routing
   back to the researcher. If sufficient (or the iteration cap is hit),
   it moves on.
5. **Reporter** (LLM + deterministic assembly) synthesizes the findings
   into a structured, multi-section report — see "Report format" below.
6. The report is saved to `output/` via the **report writer** tool.
7. **Memory writer** saves this run's queries/sources to the JSON memory
   store, for the *next* run's planner to draw on.

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
                                   reporter -> memory_writer -> END
                                    (evidence sufficient, or cap hit)
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

## Report format

`app/reporter.py` assembles the final report from two sources, never just
one:

- **Deterministic sections** — built directly from `state`, so they can
  never claim work that didn't happen: `Research Question` (the literal
  `user_goal`), `Methodology` (iteration count, tool-call counts by type,
  step count, and the critic's final `recommended_action`, all read from
  `state.tool_history`/`state.iteration`/`state.critique`), and `Sources`
  (every gathered source, de-duplicated by URL).
- **LLM-written sections** — one reporter call returns exactly four
  markdown sections (`Executive Summary`, `Key Findings`,
  `Comparison / Analysis`, `Limitations`), which are parsed out by
  heading and slotted into place. `Key Findings` is restricted to facts
  actually present in `state.findings`; interpretation/synthesis is kept
  separate in `Comparison / Analysis`, so facts and analysis are never
  mixed in one block. The LLM's `Limitations` bullets are appended after
  deterministic ones (any `state.missing_information` left over, and any
  quality `issues` the critic raised) — so a run's real gaps are always
  reported even if the LLM's own guess at limitations is thin.
- If the LLM doesn't follow the requested heading format, its whole
  reply is kept (as the Executive Summary) rather than silently dropped.
- If a run gathered no findings/sources at all, that's stated explicitly
  in `Limitations` rather than the report guessing at an answer.

Final section order: `Executive Summary → Research Question →
Methodology → Key Findings → Comparison / Analysis → Limitations →
Sources`. `app/tools/report_writer.py` writes this already-complete
report as-is (it only falls back to its old plain `# Research Report` /
`**Goal:**` wrapper for a bare, unformatted body, kept for backward
compatibility with earlier phases).

## Memory

`app/memory.py` implements the Phase 6 requirement: **previous
experience → better planning/search strategy.** This is explicitly *not*
model training and *not* a vector database — a single JSON file
(`memory/agent_memory.json` by default, path configurable via
`RESEARCHPILOT_MEMORY_PATH`) holding one small record per past run:

```json
{
  "timestamp": "2026-09-27T12:00:00+00:00",
  "goal": "What is the pricing for Acme Cloud?",
  "successful_queries": ["Acme Cloud pricing"],
  "failed_queries": ["Acme Cloud xyzzy nonsense"],
  "useful_domains": ["acme.com"],
  "iterations": 2,
  "sufficient": true
}
```

- **Writing** — the new `memory_writer` graph node runs once per
  completed research run, right after the report is generated. It reads
  `state.tool_history` to split `web_search` calls into
  `successful_queries` (returned results) and `failed_queries` (errored
  or found nothing — calculator/page_reader calls aren't queries and are
  excluded), reads `state.sources` for de-duplicated source *domains*
  (not full URLs), and appends the record to the store.
- **Retrieval** — before the planner LLM call, `app.memory.get_planning_hint`
  tokenizes the new goal (lowercased words, minus a small stopword list)
  and scores every past run's goal by keyword overlap. The top few
  overlapping runs (default 3) are rendered as a short "hints only, not
  facts" block and appended to the planner prompt — the LLM is
  explicitly told to prefer phrasings that worked before and avoid ones
  that failed, but still plan from the *current* goal. No overlap means
  no hint block is added, and the planner behaves exactly as it did
  pre-Phase-6.
- **Failure handling** — a missing or corrupt memory file is treated as
  an empty store (logged, not raised); a fresh run is never blocked by a
  broken memory file.

This intentionally does not do user-feedback storage: nothing in the
current CLI collects feedback on a finished report, and Phase 6's rule
is "do not invent functionality that is not implemented" — so that part
of the target schema is left out rather than stubbed with fake data.

> **Note on updates:** every zip/patch delivered for this project omits
> your real `memory/agent_memory.json` on purpose — it's local run
> history, not shipped code (see `.gitignore`). If you extract a new
> zip on top of this project, make sure your extraction step *merges*
> into the existing `memory/` folder rather than replacing it wholesale,
> or you'll silently lose accumulated run history (the code will just
> treat it as a fresh, empty store — no error, no crash, nothing to
> indicate anything was lost).

## Evaluation (Phase 7)

Phase 7 adds a separate deterministic evaluation layer for **completed**
research runs. It does not replace or modify the Phase 4 in-loop critic in
`app/evaluator.py`; the benchmark logic lives in `app/evaluation.py`, and
`python -m app.evaluate` runs a fixed fixture suite from
`app/evaluation_data/phase7_cases.json`.

The evaluator only uses observable artifacts (`user_goal`, `findings`,
`sources`, `iteration`, `missing_information`, and `final_report`). It never
reads or exposes hidden chain-of-thought, and it makes no LLM call by default.
The five reported 0.0-1.0 metrics are:

- **Task relevance** — fraction of meaningful task terms that appear in the
  final report. This is lexical coverage, not semantic correctness.
- **Completeness** — deterministic checks for findings, sources, executive
  summary, limitations, no unresolved `missing_information`, and termination
  within the allowed iteration limit.
- **Source coverage** — mean of URL recall and precision between collected
  `state.sources` and URLs actually cited in the report; unknown report URLs
  are flagged.
- **Grounding / evidence support** — checks `Key Findings` bullets against
  gathered findings/source metadata using a transparent lexical-overlap
  heuristic (>= 0.50). Weakly supported bullets are listed as warnings. This
  is not factual entailment or proof that a claim is true.
- **Report structure / format** — presence of all seven required Phase 5
  sections.

`overall` is the unweighted arithmetic mean of those five diagnostic scores.
No score is treated as objective ground truth. An optional LLM judge is not
needed for the current suite; if added later, it should remain a clearly
separate supplemental signal rather than overwrite deterministic metrics.

Run it with:

```bash
python -m app.evaluate
```

Each run prints a concise aggregate/per-case summary and saves the full JSON
result under `output/evaluations/evaluation_<timestamp>.json`. The fixture
dataset covers simple factual research, a multi-entity comparison, a task
requiring an additional calculation/research step, and intentionally limited
evidence.

### Phase 7 sample results

Actual output from the deterministic fixture suite in this implementation
(4 cases):

| Metric | Score |
|---|---:|
| Relevance | 1.0000 |
| Completeness | 0.9583 |
| Source coverage | 1.0000 |
| Grounding | 0.9167 |
| Format quality | 1.0000 |
| Overall | 0.9750 |

Per-case overall scores from the same run: `simple_factual=1.0000`,
`multi_entity_comparison=1.0000`, `additional_research=0.9333`, and
`limited_evidence=0.9667`. These are **fixture-suite diagnostics**, not a
live-web benchmark, not a claim that the agent improved, and not a guarantee
that future/live research will achieve the same values.

## Guardrails (Phase 8)

Phase 8 is deliberately scoped to the gaps that weren't already covered.
Most of "guardrails + failure handling" was already load-bearing
functionality from earlier phases:

| Failure mode | Where it's actually handled |
|---|---|
| API failure / provider timeout | `app/llm_provider.py` (`LLMError`, `_run_with_hard_timeout`, Groq rate-limit retry) |
| Search / page-read failure | `app/researcher.py`'s per-tool `try/except`, turned into a findings note instead of a crash |
| Malformed LLM JSON (plan or critique) | `app/planner.py` / `app/evaluator.py` (`_extract_json_array` / `_extract_json_object`, with the critic's one bounded retry) |
| Invalid calculator input | `app/tools/calculator.py` (`ast`-based evaluator, raises `CalculatorError` — never `eval`) |
| Maximum iterations | `app/evaluator.py` (`RESEARCHPILOT_MAX_ITERATIONS`, enforced before any further LLM call) |
| Missing sources / unsupported claims | `app/reporter.py`'s grounding gate (skips narrative synthesis when the critic didn't confirm sufficiency) and deterministic Limitations bullets |
| Empty/failed report synthesis | `app/reporter.py`'s deterministic fallback narrative (evidence preserved, no invented conclusions) |

What Phase 8 actually added, in `app/guardrails.py`:

- **Input validation** (`validate_task`, used by `app/main.py` before the
  graph is ever invoked): rejects an empty/whitespace-only task, a task
  over 2000 characters (malformed/abusive input that would just blow up
  every downstream prompt budget for no benefit), and a task with no
  actual word characters at all (e.g. `"??? !!!"`) — not a real research
  question in any recognizable sense.
- **Outbound URL validation** (`validate_fetch_url`, wired into
  `app/tools/page_reader.py::read_page`): `page_reader` is the one tool
  argument in the system that is LLM/search-result influenced *and*
  reaches outside the process (the calculator only evaluates arithmetic;
  web_search only takes a query string), so it's the one real SSRF
  surface. Blocked: any scheme other than `http`/`https` (e.g.
  `file:///etc/passwd`), a URL with no host, and a host that is a literal
  loopback/private/link-local/reserved/multicast IP or a known-local
  hostname (`localhost`, cloud-metadata hostnames, etc.). Deliberately
  does **not** perform DNS resolution — that would make the check a real
  network call before `requests.get` even runs, and would break this
  project's "tools are tested against canned responses, no real network
  access" convention. It catches the common, cheap SSRF cases rather than
  claiming to be a complete defense against DNS rebinding.
- A defensive empty-report check in `app/main.py`, right before saving:
  every current path through `app/reporter.py` already emits a
  fully-headed report (with `"(no findings gathered)"`-style
  placeholders when evidence is thin), so this should be unreachable —
  it exists so a truly blank report is never silently written to disk.

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
│   ├── evaluator.py      # Phase 4 in-loop coverage + quality critic / router
│   ├── evaluation.py     # NEW (Phase 7): deterministic completed-run metrics
│   ├── evaluate.py       # NEW (Phase 7): `python -m app.evaluate` suite CLI
│   ├── evaluation_data/
│   │   └── phase7_cases.json  # representative fixed evaluation fixtures
│   ├── reporter.py       # structured multi-section report (see "Report
│   │                     #   format" above), not just a plain answer
│   ├── memory.py         # NEW (Phase 6): JSON-backed run history +
│   │                     #   keyword-overlap retrieval (see "Memory" above)
│   ├── guardrails.py     # NEW (Phase 8): input task validation +
│   │                     #   page_reader URL/SSRF validation (see "Guardrails" above)
│   └── tools/
│       ├── web_search.py
│       ├── page_reader.py    # fetch + extract text from one URL
│       ├── calculator.py     # wired into the agent loop
│       └── report_writer.py  # save the final report to output/ (backward-
│                              #   compatible with a bare, unformatted body)
├── tests/
├── output/                # generated reports land here
│   └── evaluations/       # Phase 7 timestamped evaluation JSON results
├── memory/                 # NEW (Phase 6): agent_memory.json lands here
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
| `LLM_TIMEOUT_SECONDS` | Maximum duration for one provider request before it fails cleanly (default `60`). Prevents a stalled LLM request from hanging the agent indefinitely. |
| `GROQ_API_KEY` | Your Groq API key. Required when `LLM_PROVIDER=groq`. Never commit this. |
| `GROQ_REQUESTS_PER_MINUTE` | Client-side throttle for Groq calls (default `25`). See "Rate limiting" below. |
| `GROQ_REASONING_EFFORT` | `low`/`medium`/`high` reasoning budget, sent only for GPT-OSS models (`openai/gpt-oss-20b`/`-120b`; ignored for other Groq models). Default `low`. See "GPT-OSS empty responses" below — this is what fixes empty critic/reporter output on those models. |
| `ANTHROPIC_API_KEY` | Your Anthropic API key. Only required when `LLM_PROVIDER=anthropic`. |
| `RESEARCHPILOT_MAX_ITERATIONS` | Max research/evaluate loop iterations before forcing a report (default `3`). Prevents infinite loops. |
| `RESEARCHPILOT_MEMORY_PATH` | Path to the JSON memory store (default `memory/agent_memory.json`). Blank/unset values safely fall back to that default. See "Memory" above. |
| `RESEARCHPILOT_EVALUATOR_TIMEOUT_SECONDS` | Critic wall-clock ceiling (default `45`). On the first critic timeout, the agent performs one deterministic deeper-source recovery pass using already-collected URLs; a later timeout ends research and reports the uncertainty explicitly. |
| `RESEARCHPILOT_REPORTER_TIMEOUT_SECONDS` | Final report LLM wall-clock ceiling (default `45`). If synthesis exceeds it, ResearchPilot immediately produces a deterministic evidence-only fallback report instead of hanging. If the critic explicitly did not confirm evidence sufficiency, LLM narrative synthesis is skipped entirely and an evidence-only report is produced. |
| `RESEARCHPILOT_REPORTER_MAX_TOKENS` | Maximum tokens requested for final narrative synthesis (default `1000`). |
| `RESEARCHPILOT_EVALUATOR_MAX_TOKENS` | Maximum critic JSON generation budget (default `900`). Kept bounded because the critic returns structured JSON only. |

### Rate limiting (Groq)

Provider calls also have a bounded request timeout (`LLM_TIMEOUT_SECONDS`, default 60 seconds), so a stalled API request returns control instead of hanging forever. The in-loop critic uses its own 45-second ceiling and one bounded recovery pass described below.

Two Groq rate-limit layers, both in `app/llm_provider.py`:

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

### GPT-OSS empty responses (`openai/gpt-oss-20b` / `-120b`)

These are *reasoning* models: on Groq's chat-completions endpoint, reasoning
tokens and the final answer share the same `max_tokens` budget. If reasoning
consumes the whole budget, `message.content` comes back as an **empty
string with no error at all** — indistinguishable, from the caller's side,
from the model simply choosing to say nothing. This is what
`evaluator.py`'s bounded critic retry (`"No JSON object found in evaluator
output: ''"`) was built to catch, and it correctly falls back to an
evidence-only report rather than guessing — but it's still worth avoiding,
since it means no LLM-synthesized answer that pass.

`GroqClient` now sends `reasoning_effort=low` (via `GROQ_REASONING_EFFORT`,
default `low`) for GPT-OSS models specifically — Groq only accepts this
field for GPT-OSS 20B/120B, so it's omitted for every other model. This
caps how much of the budget reasoning is allowed to spend, leaving room for
the actual JSON/text answer. It's most likely to matter on the smaller
`openai/gpt-oss-20b` and on the critic/reporter's larger prompts (more
gathered findings = more for the model to reason about before answering).
If empty responses persist even at `low`, raising
`RESEARCHPILOT_EVALUATOR_MAX_TOKENS` / `RESEARCHPILOT_REPORTER_MAX_TOKENS`
gives the model more total room to fit both reasoning and the answer.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

`tests/test_mvp.py` (Phase 1), `tests/test_phase2.py` (Phase 2),
`tests/test_phase3.py` (Phase 3), `tests/test_phase4.py` (Phase 4),
`tests/test_phase5.py` (Phase 5), `tests/test_phase6.py` (Phase 6),
`tests/test_phase7.py` (Phase 7), and `tests/test_phase8.py` (Phase 8)
cover `AgentState`, all four tools (`calculator`, `page_reader`,
`report_writer`, and `web_search` indirectly via the researcher),
JSON-extraction (planner + evaluator), the evaluator/critic's routing
decision, iteration cap, and quality-`issues` surfacing,
`tool_selector.choose_tool` for every step shape, the researcher's dedup
and tool-routing logic, `generate_report`'s section parsing/assembly
(order, deterministic Methodology/Sources/Research Question content, the
no-findings and malformed-LLM-output fallbacks), and `app/memory.py`'s
store load/save/corrupt-file handling, `record_run`'s
successful/failed-query split and domain de-duplication,
`retrieve_relevant_experience`'s keyword-overlap matching, and the
planner's inclusion (or graceful omission) of a memory hint in its
prompt, plus Phase 7 evaluation-result creation, required-section checks,
source coverage, grounding warnings, incomplete research, and iteration-limit
behavior, plus Phase 8's `validate_task` (empty/whitespace/too-long/
no-word-character rejection), `validate_fetch_url` (blocked schemes,
loopback/private/link-local/known-local-hostname rejection), and
`read_page` refusing a blocked URL without ever reaching `requests.get`
— all without needing network access or an API key
(`requests.get` / `web_search` / `read_page` / the LLM client are
monkeypatched out wherever a test would otherwise need the network;
`write_report` and the memory store's file I/O are exercised for real
against a `tmp_path`, since both are pure filesystem ops safe to run
anywhere). They do **not** cover live LLM calls or live web search
themselves, since those need real credentials and open internet access;
verify those manually with `python -m app.main "..."` (see "How this was
tested" below).

## How this was tested

### Phase 8

Executed in the same environment as Phase 7, with only `app/guardrails.py`,
`app/tools/page_reader.py`, and `app/main.py` changed:
- `pytest tests/ -v`: **113/113 passed** (all Phase 1-7 regression tests
  unchanged, plus 16 new Phase 8 tests covering task validation, URL/SSRF
  validation, and `page_reader` refusing a blocked URL before any HTTP call
  is attempted).
- Manual CLI check of the three input-guardrail paths (`python -m app.main`
  with an empty string, a 3000-character string, and `"??? !!!"`) — each
  exits with status `1` and a clear one-line error, before any LLM call or
  network access is attempted.
- No live web or LLM call was needed for any Phase 8 change or test.

### Phase 7

Executed in the Phase 7 build environment with the real installed project
dependencies:
- `pytest tests/ -v`: **97/97 passed** after the final Phase 7 stabilization patch. This includes all Phase 1-7 regression tests plus coverage for provider hard timeouts, blank memory-path handling, bounded critic prompts, one-shot timeout recovery, second-timeout termination, empty/malformed reporter output, reporter hard timeouts, and the grounding gate that blocks narrative synthesis when critic sufficiency is unconfirmed.
- `python -m app.evaluate`: **4 fixture cases executed** and a timestamped
  JSON result was written to `output/evaluations/`. Aggregate scores were
  relevance `1.0000`, completeness `0.9583`, source coverage `1.0000`,
  grounding `0.9167`, format quality `1.0000`, overall `0.9750`.
- No live web or LLM call is part of the default Phase 7 suite, deliberately,
  so rerunning it is reproducible and does not depend on changing search
  results, provider nondeterminism, credentials, or rate limits. Live agent
  calls are separately protected by `LLM_TIMEOUT_SECONDS` (default 60s).

### Phase 6

Confirmed on the user's own machine (Windows, Python 3.14.7,
`pytest-9.1.1`, real `pydantic`/`langgraph`/`anthropic`/`groq`
installed), superseding the offline-shimmed checks described below:
- `python -m pytest tests/ -v`: **79/79 passed**, including all 20
  `tests/test_phase6.py` tests, with zero regressions in
  `test_mvp.py`/`test_phase2-5.py`.
- A live `python -m app.main "What is the current pricing for
  Anthropic's Claude API, including the newest models?"` run confirmed
  the whole Phase 6 wiring end-to-end against real network + LLM calls:
  - `[MEMORY] No related past runs found.` on this first-ever run (empty
    store), and `[MEMORY] Saved this run's queries/sources for future
    planning.` after the report — `memory/agent_memory.json` now holds
    one record. A second, related run of the same question then logged
    `[MEMORY] Found 1 related past run(s) — adding hints to the plan
    prompt.`, confirming `retrieve_relevant_experience`'s keyword-overlap
    match actually fires against a real on-disk store, not just the
    offline harness below — closing the one gap noted after the first
    live run.
  - The `reporter -> memory_writer -> END` edge ran cleanly as part of a
    real `build_graph()` invocation — not just something exercised via a
    dataclass stand-in.
  - The critic/evaluator loop ran for the full 4 iterations (3 more
    passes + the max-iteration cutoff), each time correctly flagging
    quality issues — reliance on third-party price-aggregator sources
    (`llmpricecheck.com`, `pricepertoken.com`, `coursiv.io`,
    `claudelab.net`), and unverified/undated figures — without those
    issues alone forcing endless re-search once coverage looked
    sufficient. This is exactly the documented Phase 5 limitation in
    practice: *"the critic doesn't yet re-route research specifically to
    resolve a quality issue that isn't also a coverage gap"* — worth
    keeping in mind when reading a report's numbers, since a flagged
    quality issue doesn't block the report from citing that source.

What was *not* re-verified by these real runs (still only checked via
the offline harness below, or not applicable to these queries): the
missing-file/corrupt-JSON fallback paths in `load_memory` (no corrupt
file occurred during a normal run).

<details>
<summary>Original offline validation (before the real run above)</summary>

This environment (used to build Phase 6) had no network access
(`pip install -r requirements.txt` could not run — no `pydantic`,
`langgraph`, etc.) and no LLM/API credentials. What was actually
*executed* there:
- `python -m py_compile` on every touched file (`app/memory.py`,
  `app/planner.py`, `app/graph.py`, and the new `tests/test_phase6.py`):
  clean.
- `app/memory.py`'s real code — `load_memory`/`save_memory` (including
  the missing-file and corrupt-JSON fallback paths), `record_run` (the
  successful/failed `web_search`-query split, domain de-duplication from
  `state.sources`, the `sufficient` default when `state.critique` is
  `None`), `retrieve_relevant_experience` (keyword-overlap scoring and
  the `max_runs` cap), `format_memory_hint`, and `get_planning_hint` —
  was run directly against real temp-file paths via a standalone
  harness, using a small dataclass-based stand-in for `app.state`'s
  `AgentState`/`Source`/`ToolCallRecord` (since real `pydantic` wasn't
  installable offline there, the same constraint Phase 5 hit). Every
  assertion from `tests/test_phase6.py`'s equivalent scenarios was
  checked this way and passed, including the "persists across separate
  record/retrieve calls" case.
- `app/planner.py`'s actual `plan()` function was exercised the same way
  (real `app.planner` module, `app.state`/`app.llm_provider` stubbed),
  confirming the memory hint text is folded into the LLM prompt when a
  related past run exists, and that the prompt is byte-identical to the
  pre-Phase-6 form (`"Research goal: {goal}"`, no trailing hint section)
  when nothing relevant is found.

</details>

### Phase 5

The environment used to build Phase 5 has no network access (so
`pip install -r requirements.txt` — `langgraph`, `pydantic`, etc. — could
not run) and no LLM/API credentials. Given that, what was actually
*executed* here:
- `app/reporter.py`'s pure logic (`_split_sections`, `_build_methodology`,
  `_build_limitations`, `_build_sources`) was run directly with a small
  standalone harness (no `app.state`/pydantic dependency), covering
  heading parsing, tool-count formatting, and URL de-duplication.
- `tests/test_phase5.py` (7 new tests: section parsing with/without valid
  headings, full `generate_report` assembly against a `FakeLLM` including
  section order and deterministic-content checks, the no-findings case,
  the malformed-LLM-output fallback, and both `write_report` behaviors —
  passthrough for an already-formatted report and the old wrap-a-bare-body
  path for backward compatibility) was executed against the real
  `app/reporter.py` and `app/tools/report_writer.py` using minimal
  same-shape shims for `pydantic.BaseModel`/`Field` and `dotenv` (since
  those packages themselves aren't installable offline here) — **all 7
  passed**. `tests/test_phase3.py`'s two existing `write_report` tests
  were re-run the same way to confirm no regression — **both passed**.
- A full sample report was rendered end-to-end (`generate_report` with a
  scripted `FakeLLM`, real `Source`/`ToolCallRecord` objects) to eyeball
  the actual output — section order, methodology numbers, and source
  de-duplication all matched expectations.
- **Not** run here: `python -m pytest tests/ -v` against the *real*
  `pydantic`/`langgraph` install, `python -m py_compile`, and any live
  `python -m app.main "..."` run. **Run these yourself** — with
  `requirements.txt` installed and `.env` filled in — before treating
  Phase 5 as fully validated; the shimmed tests prove `app/reporter.py`'s
  logic is correct, not that it imports cleanly against the real
  dependency versions pinned in `requirements.txt`.

### Phase 3 (unchanged since)

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
treating Phase 3 (or Phase 5's report formatting on real output) as fully
validated end-to-end — the mocked test proves
the wiring is correct, not that DuckDuckGo's current HTML markup or a
particular live page still parses as expected. A good test query is one
that plausibly needs a calculation and a deeper look at one source, e.g.
"What would a year of the Notion paid plan cost, and how does that
compare to the free plan's limits?", so you can watch [TOOL] web_search,
[TOOL] calculator, and potentially [TOOL] page_reader all fire in one
run.

## Known limitations (Phase 8 stage, expected)

- `validate_fetch_url` blocks known-local hostnames and literal
  loopback/private/link-local/reserved IPs, but does **not** resolve
  hostnames via DNS — so it does not defend against DNS rebinding (a
  hostname that resolves to a public IP at validation time but a private
  one at request time). This was a deliberate tradeoff to keep the check
  pure/offline, consistent with this project's tool-testing convention.
- Task input validation (`validate_task`) is a small set of deterministic
  length/shape checks, not an LLM-based "is this actually a sensible
  research question" classifier — per Phase 8's own "do not over-engineer
  safety features unrelated to the contest" instruction.
- Guardrails only cover the two gaps that weren't already handled
  elsewhere (see the "Guardrails" table above); they don't change any of
  the existing timeout/retry/fallback behavior from Phases 2-7.

- The critic's quality `issues` (unsupported claims, off-topic sources)
  are surfaced in the report's Limitations section, but don't otherwise
  change agent behavior beyond what Phase 2's coverage check already
  drove — the critic doesn't yet re-route research specifically to
  resolve a quality issue that isn't also a coverage gap.
- When the critic confirms sufficiency, `Key Findings` vs `Comparison / Analysis` still depend on the reporter LLM following the requested section semantics. When the critic explicitly does **not** confirm sufficiency, narrative synthesis is skipped and the deterministic evidence-only fallback is used instead.
- If the evaluator's LLM call fails, the code does **not** mark evidence sufficient. A first timeout can trigger one bounded deeper-source recovery pass. If sufficiency still cannot be confirmed, the reporter skips LLM narrative synthesis and emits an evidence-only report so unsupported claims are not promoted into confident conclusions.
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
- Memory retrieval is plain keyword overlap on goal text, not semantic
  similarity — a related goal phrased with entirely different words
  (e.g. "Notion cost" vs. "How much do I pay for a workspace tool") won't
  match. No embeddings/vector store, per Phase 6's "do not build a
  complex vector database unless actually necessary."
- Memory only records `web_search` queries as successful/failed;
  `page_reader`/`calculator` calls aren't queries and aren't stored as
  search patterns, so a run driven mostly by page-reading leaves fewer
  hints for next time.
- No user-feedback storage — the CLI doesn't currently collect feedback
  on a finished report, so that part of the Phase 6 schema is left out
  rather than faked (see "Memory" above).
- The memory store is a single flat JSON file with no size cap or
  pruning — fine at prototype/contest scale, but it will grow unbounded
  over many real runs.
- Phase 7's default benchmark uses fixed completed-run fixtures rather than
  executing live research. It is useful for regression and transparency, but
  it does not measure current web-search quality or provider behavior.
- Relevance and grounding are lexical heuristics. Synonyms/paraphrases can be
  under-scored, while lexical overlap can over-score a claim that is phrased
  similarly but is still wrong; the evaluator therefore flags likely issues
  rather than claiming semantic truth.
- The four-case dataset is intentionally small and representative, not a
  statistically meaningful benchmark. Scores should not be used to claim the
  agent improved without a larger controlled comparison across versions.

### Live-run timeout behavior

Provider SDK calls are protected by `LLM_TIMEOUT_SECONDS` (default 60 seconds),
and the critic has a stricter `RESEARCHPILOT_EVALUATOR_TIMEOUT_SECONDS`
(default 45 seconds). If the first critic request exceeds that wall-clock
limit, ResearchPilot does **not** mark the evidence sufficient. Instead it
deterministically selects up to three already-collected, unread source URLs
and performs one deeper `page_reader` recovery pass before running the critic
again. If the critic is still unavailable after that bounded recovery, the
loop stops and the report explicitly records that evidence sufficiency could
not be confirmed. In that state the reporter **does not call the narrative
LLM at all**; it emits a deterministic evidence-only report from the findings
and sources already stored in state. This prevents an unverified synthesis
from inventing prices, dates, model names, or other conclusions. Timeout
failures are not retried immediately, while fast malformed/empty critic
responses may still retry once. Critic and report prompts also retain only a
bounded amount of the newest gathered evidence so later research passes do
not grow the LLM input without limit.
