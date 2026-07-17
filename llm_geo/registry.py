"""Adapter: bridges the ground-truth operation registry (`llm_geo/operations/registry.py`,
populated by `llm_geo/tools/public_data_providers.py`) into the id -> {kind, description,
inputs, outputs, fn} shape the planner/executor expect.

Each ground-truth operation returns a single concrete value (`RegisteredOperation.output_type`).
This maps that single value onto the named-output-dict convention (`run` returning a dict) the
rest of the pipeline (and LLM-generated custom nodes) use, via `_OUTPUT_NAME_BY_TYPE`.
"""
from __future__ import annotations

from typing import Any

from .operations import load_all_operations
from .operations.registry import RegisteredOperation, registered_operations

_OUTPUT_NAME_BY_TYPE = {"GeoDataFrame": "features", "dict": "report"}


def _adapt(operation: RegisteredOperation) -> dict[str, Any]:
    inputs = {name: type_name for name, type_name, _ in operation.inputs}
    output_name = _OUTPUT_NAME_BY_TYPE.get(operation.output_type, "value")

    def fn(**kwargs: Any) -> dict[str, Any]:
        call_kwargs = {name: kwargs[name] for name in inputs if name in kwargs}
        return {output_name: operation.function(**call_kwargs)}

    return {
        "kind": operation.kind,
        "description": operation.description,
        "inputs": inputs,
        "outputs": {output_name: operation.output_type},
        "fn": fn,
    }


load_all_operations()
REGISTRY: dict[str, dict[str, Any]] = {op.id: _adapt(op) for op in registered_operations()}


def catalog_text() -> str:
    """Registry catalog for the planner prompt, rendered in the same name (type): description
    port format the planner must emit for custom nodes."""
    lines = []
    for operation in registered_operations():
        output_name = _OUTPUT_NAME_BY_TYPE.get(operation.output_type, "value")
        inputs = "; ".join(
            f"{name} ({type_name}"
            + (f", default {operation.defaults[name]!r}" if name in operation.defaults else "")
            + f"): {description}"
            for name, type_name, description in operation.inputs
        ) or "(none)"
        lines.append(
            f"- {operation.id} [{operation.kind}]: {operation.description}\n"
            f"    inputs: {inputs}\n"
            f"    output: {output_name} ({operation.output_type}): {operation.output_description}"
        )
    return "\n".join(lines)
