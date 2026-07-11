"""Probe the configured model endpoint's structured-output capabilities."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Literal

from main import initialize_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm_geo.subagents.runtime import ask_structured, create_structured_agent
STRATEGIES = ("auto", "tool", "provider", "json_mode", "prompted")


class ProbeResult(BaseModel):
    """Exact result returned by the structured-output capability probe."""

    status: Literal["ok"]
    value: int = Field(description="The requested integer value")
    source: Literal["prompt", "tool"]


@tool
def structured_output_probe_value() -> int:
    """Return the integer required by the structured-output probe."""
    return 37


def run_probe(model, strategy: str, with_tool: bool) -> tuple[bool, str, float]:
    os.environ["LLM_GEO_STRUCTURED_OUTPUT"] = strategy
    tools = [structured_output_probe_value] if with_tool else []
    expected_source = "tool" if with_tool else "prompt"
    prompt = (
        "Call structured_output_probe_value, then return status='ok', its integer "
        "as value, and source='tool'. Do not invent the value."
        if with_tool
        else "Return status='ok', value=23, and source='prompt'."
    )
    started = time.perf_counter()
    try:
        agent = create_structured_agent(
            model,
            "Follow the request exactly and do not add unsupported values.",
            ProbeResult,
            tools=tools,
        )
        result = ask_structured(agent, prompt)
        expected_value = 37 if with_tool else 23
        if result.value != expected_value or result.source != expected_source:
            raise AssertionError(
                f"unexpected result: {result.model_dump_json()}"
            )
        return True, result.model_dump_json(), time.perf_counter() - started
    except Exception as error:  # Diagnostic must continue through every strategy.
        detail = f"{type(error).__name__}: {error}"
        return False, detail[:1000], time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, action="append")
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of repetitions per strategy/tool configuration (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("structured_output_benchmark_results.json"),
        help="JSON results file",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    load_dotenv()
    model = initialize_model()
    strategies = tuple(args.strategy or STRATEGIES)
    print(f"Runs per configuration: {args.runs}")
    print()
    total_failures = 0
    configurations: list[dict[str, object]] = []
    for with_tool in (False, True):
        print("WITH TOOL" if with_tool else "WITHOUT TOOL")
        for strategy in strategies:
            passes = 0
            durations: list[float] = []
            failure_details: Counter[str] = Counter()
            for run_number in range(1, args.runs + 1):
                passed, detail, duration = run_probe(model, strategy, with_tool)
                passes += int(passed)
                durations.append(duration)
                if not passed:
                    failure_details[detail] += 1
                if args.runs >= 20 and (
                    run_number % 10 == 0 or run_number == args.runs
                ):
                    print(
                        f"  {strategy:<10} progress {run_number:>3}/{args.runs}",
                        end="\r",
                        flush=True,
                    )
            failures = args.runs - passes
            total_failures += failures
            pass_rate = passes / args.runs
            mean_duration = statistics.fmean(durations)
            print(
                f"  {strategy:<10} pass={passes:>3} fail={failures:>3} "
                f"rate={pass_rate:6.1%} mean={mean_duration:6.2f}s"
            )
            common_failures = failure_details.most_common(3)
            for detail, count in common_failures:
                print(f"    {count:>3}x {detail}")
            configurations.append(
                {
                    "strategy": strategy,
                    "with_tool": with_tool,
                    "runs": args.runs,
                    "passes": passes,
                    "failures": failures,
                    "pass_rate": pass_rate,
                    "duration_seconds": {
                        "mean": mean_duration,
                        "median": statistics.median(durations),
                        "min": min(durations),
                        "max": max(durations),
                    },
                    "failure_examples": [
                        {"count": count, "detail": detail}
                        for detail, count in common_failures
                    ],
                }
            )
        print()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs_per_configuration": args.runs,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Results written to {args.output.resolve()}")
    return int(total_failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
