"""Run with: poetry run python -m llm_geo.cli "<geo-analysis task>" """
from __future__ import annotations

import sys

from dotenv import load_dotenv

from .graph import run


def main() -> None:
    load_dotenv()
    task = " ".join(sys.argv[1:]) or "Buffer sample points by 100m and summarize the result."
    report = run(task)
    result = report.result
    print("\nSUCCESS" if result.success else f"\nFAILED: {result.error}")
    for node_id, output in result.outputs.items():
        print(f"- {node_id}: {list(output.keys())}")
    print(f"\nDebug bundle (prompts, code attempts, errors, full report): {report.artifacts_dir}")


if __name__ == "__main__":
    main()
