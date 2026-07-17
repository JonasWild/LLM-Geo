"""Adapter: bridges the ground-truth operation registry (`llm_geo/operations/registry.py`,
populated by `llm_geo/tools/public_data_providers.py`) into the id -> {kind, description,
inputs, outputs, fn} shape the planner/executor expect.

Each ground-truth operation returns a single concrete value (`RegisteredOperation.output_type`).
This maps that single value onto the named-output-dict convention `run(**inputs) -> dict` the
rest of the pipeline (and LLM-generated custom nodes) use, via `_OUTPUT_NAME_BY_TYPE`.
"""
from __future__ import annotations

from typing import Any

from .operations.registry import RegisteredOperation, registered_operations
from .tools import public_data_providers  # noqa: F401  (side effect: registers @code operations)

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
        "defaults": operation.defaults,
        "outputs": {output_name: operation.output_type},
        "fn": fn,
    }


REGISTRY: dict[str, dict[str, Any]] = {op.id: _adapt(op) for op in registered_operations()}


def catalog_text() -> str:
    lines = []
    for op in registered_operations():
        inputs = ", ".join(
            f"{name}: {type_name} ({description})"
            + (f" [default={op.defaults[name]!r}]" if name in op.defaults else "")
            for name, type_name, description in op.inputs
        )
        output_name = _OUTPUT_NAME_BY_TYPE.get(op.output_type, "value")
        lines.append(
            f"- {op.id} [{op.kind}]: {op.description} "
            f"inputs=({inputs}) output={output_name}: {op.output_type} ({op.output_description})"
        )
    return "\n".join(lines)
