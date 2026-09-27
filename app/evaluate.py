"""CLI for the deterministic Phase 7 ResearchPilot evaluation suite.

Usage:
    python -m app.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation import evaluate_run
from app.state import AgentState

DATASET_PATH = Path(__file__).with_name("evaluation_data") / "phase7_cases.json"
OUTPUT_DIR = Path("output/evaluations")


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    return data


def run_suite(path: Path = DATASET_PATH) -> dict:
    cases = load_dataset(path)
    results = []

    for case in cases:
        state = AgentState(**case["run"])
        result = evaluate_run(state, allowed_iterations=int(case.get("allowed_iterations", 3)))
        results.append({
            "id": case["id"],
            "category": case["category"],
            "task": state.user_goal,
            "result": result.model_dump(),
        })

    metrics = ("relevance", "completeness", "source_coverage", "grounding", "format_quality", "overall")
    aggregate = {
        metric: round(sum(item["result"][metric] for item in results) / len(results), 4)
        for metric in metrics
    } if results else {metric: 0.0 for metric in metrics}

    return {
        "suite": "phase7_deterministic_fixture_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "aggregate": aggregate,
        "cases": results,
    }


def save_results(payload: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"evaluation_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    payload = run_suite()
    path = save_results(payload)

    print(f"Evaluation suite: {payload['suite']}")
    print(f"Cases: {payload['case_count']}")
    print("Aggregate deterministic scores (0.0-1.0):")
    for name, value in payload["aggregate"].items():
        print(f"  {name:16} {value:.4f}")
    print("\nPer-case overall:")
    for case in payload["cases"]:
        print(f"  {case['id']:24} {case['result']['overall']:.4f}  {case['category']}")
    print(f"\nSaved: {path}")
    print("Note: fixture-suite scores are deterministic diagnostics, not objective ground truth or a live-web benchmark.")


if __name__ == "__main__":
    main()
