"""
Report file writer tool.

Saves a finished report to output/ with a timestamped filename. This was
inline in `app/main.py` in Phase 1/2; pulling it out here means it's an
independently testable tool like the others, and it's the thing the
"completed research -> file writer" branch of tool selection refers to.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def write_report(user_goal: str, content: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Write `content` (the synthesized report body) to a timestamped
    Markdown file under `output_dir`, prefixed with the goal. Returns the
    path written to. Raises OSError on a filesystem failure (e.g.
    unwritable directory).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"report_{timestamp}.md"
    path.write_text(
        f"# Research Report\n\n**Goal:** {user_goal}\n\n{content}\n",
        encoding="utf-8",
    )
    return path
