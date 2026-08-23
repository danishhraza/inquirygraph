"""Run the small benchmark and report simple, transparent metrics."""

import json
from pathlib import Path

from inquirygraph.agent.graph import run_investigation


def score_report(report, case: dict) -> dict:
    text = " ".join([report.executive_summary] + [section.content for section in report.sections]).lower()
    expected = [term.lower() for term in case.get("must_include", [])]
    covered = sum(term in text for term in expected)
    citations = sum(bool(section.citations) for section in report.sections)
    return {
        "id": case["id"],
        "theme_recall": covered / len(expected) if expected else 1.0,
        "sections_with_citations": citations,
    }


def main() -> None:
    cases = json.loads((Path(__file__).parent / "benchmark.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        state = run_investigation(case["id"], case["query"])
        report = state.get("final_report")
        results.append(score_report(report, case) if report else {"id": case["id"], "error": state.get("errors")})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()