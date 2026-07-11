"""Probe the configured model endpoint's structured-output capabilities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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


def initialize_model():
    model_name = os.getenv("LLM_GEO_MODEL", "gpt-5.4-mini").strip()
    if not model_name:
        raise RuntimeError("LLM_GEO_MODEL is empty")
    provider = os.getenv("LLM_GEO_MODEL_PROVIDER", "openai").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    options: dict[str, str | bool] = {}
    if provider:
        options["model_provider"] = provider
    if base_url:
        options["base_url"] = base_url
        options["use_responses_api"] = False
    return model_name, provider, base_url, init_chat_model(model_name, **options)


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
    args = parser.parse_args()
    load_dotenv()
    model_name, provider, base_url, model = initialize_model()
    strategies = tuple(args.strategy or STRATEGIES)
    print(f"Model: {model_name}")
    print(f"Provider: {provider or 'inferred'}")
    print(f"Endpoint: {base_url or 'provider default'}")
    print()
    failures = 0
    for with_tool in (False, True):
        print("WITH TOOL" if with_tool else "WITHOUT TOOL")
        for strategy in strategies:
            passed, detail, duration = run_probe(model, strategy, with_tool)
            failures += not passed
            print(
                f"  {strategy:<10} {'PASS' if passed else 'FAIL':<4} "
                f"{duration:6.2f}s  {detail}"
            )
        print()
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
