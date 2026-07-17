"""Trusted, typed Python operations available to LLM-GEO workflows."""

from importlib import import_module
from pkgutil import walk_packages

from llm_geo.operations.registry import RegisteredOperation, code, registered_operations


def load_all_operations() -> tuple[RegisteredOperation, ...]:
    """Import every operations module and return all registered functions."""
    prefix = f"{__name__}."
    for module in walk_packages(__path__, prefix):
        import_module(module.name)
    return registered_operations()


__all__ = [
    "RegisteredOperation",
    "code",
    "load_all_operations",
    "registered_operations",
]
