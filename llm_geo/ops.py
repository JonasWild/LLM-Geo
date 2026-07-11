"""Flat, lazy-loaded public namespace for trusted workflow operations."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    from llm_geo.operations import load_all_operations

    operations = load_all_operations()
    for operation in operations:
        if operation.name == name:
            globals()[name] = operation.function
            return operation.function
    raise AttributeError(f"module {__name__!r} has no operation {name!r}")


def __dir__() -> list[str]:
    from llm_geo.operations import load_all_operations

    return sorted({*globals(), *(operation.name for operation in load_all_operations())})
