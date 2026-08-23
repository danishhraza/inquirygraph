"""CLI helper to run one investigation without curl."""

import json
import sys

from inquirygraph.agent.graph import run_investigation
from inquirygraph.config.settings import settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_investigation.py \"Your research question\"")
        sys.exit(1)

    if not settings.openrouter_api_key:
        print("Set OPENROUTER_API_KEY in .env first.")
        sys.exit(1)

    query = sys.argv[1]
    import uuid

    inv_id = str(uuid.uuid4())
    print(f"Running investigation {inv_id}...\n")

    result = run_investigation(inv_id, query)
    report = result.get("final_report")

    print(f"LLM calls: {result.get('llm_call_count')}")
    print(f"Iterations: {result.get('iteration')}")
    print(f"Errors: {result.get('errors')}\n")

    if report:
        print("=== EXECUTIVE SUMMARY ===")
        print(report.executive_summary)
        for section in report.sections:
            print(f"\n=== {section.title} ===")
            print(section.content)
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
