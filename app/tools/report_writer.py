"""
Report file writer tool.

Saves a finished report to output/ with a timestamped filename. This was
inline in `app/main.py` in Phase 1/2; pulling it out here means it's an
independently testable tool like the others, and it's the thing the
"completed research -> file writer" branch of tool selection refers to.

Phase 5 change: `app.reporter.generate_report` now produces a complete,
already-formatted report (its own "# Research Report" header, a
"## Research Question" section containing the goal, etc. - see
app/reporter.py). This writer no longer needs to add its own header in
that case; it only falls back to the old "# Research Report / **Goal:**"
wrapper for callers that still pass a bare, unformatted body (kept for
backward compatibility - see tests/test_phase3.py).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def write_report(user_goal: str, content: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Write `content` to a timestamped Markdown file under `output_dir`.
    Returns the path written to. Raises OSError on a filesystem failure
    (e.g. unwritable directory).

    If `content` is already a complete report (starts with the
    "# Research Report" header - the case for every report produced by
    the current `app.reporter.generate_report`), it is written as-is.
    Otherwise it's wrapped with a minimal header, same as Phase 1-4.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"report_{timestamp}.md"

    if content.lstrip().startswith("# Research Report"):
        text = content if content.endswith("\n") else content + "\n"
    else:
        text = f"# Research Report\n\n**Goal:** {user_goal}\n\n{content}\n"

    path.write_text(text, encoding="utf-8")
    return path
