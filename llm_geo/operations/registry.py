"""Strict registration for trusted, prewritten workflow operations."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable, Literal, get_type_hints, overload


@dataclass(frozen=True)
class RegisteredOperation:
    """A trusted top-level function and its planner-visible contract."""

    id: str
    function: Callable[..., object]
    module: str
    name: str
    description: str
    inputs: tuple[tuple[str, str, str], ...]
    defaults: dict[str, Any]
    output_type: str
    output_description: str
    category: Literal["retrieval", "transformation"] = "transformation"

    @property
    def qualified_id(self) -> str:
        """Return the legacy source-qualified operation identifier."""
        return f"{self.module}.{self.name}"

    @property
    def import_statement(self) -> str:
        """Return the stable public import exposed to generated programs."""
        return f"from llm_geo.ops import {self.name}"

    def catalog_entry(self) -> dict[str, object]:
        return {
            "id": self.id,
            "import": self.import_statement,
            "category": self.category,
            "description": self.description,
            "inputs": [
                {
                    "name": name,
                    "type": annotation,
                    "description": description,
                    "required": name not in self.defaults,
                    **({"default": self.defaults[name]} if name in self.defaults else {}),
                }
                for name, annotation, description in self.inputs
            ],
            "output": {
                "type": self.output_type,
                "description": self.output_description,
            },
        }


_OPERATIONS: dict[str, RegisteredOperation] = {}


@overload
def code(function: Callable[..., object]) -> Callable[..., object]: ...


@overload
def code(
    function: None = None,
    *,
    category: Literal["retrieval", "transformation"] = "transformation",
) -> Callable[[Callable[..., object]], Callable[..., object]]: ...


def code(
    function: Callable[..., object] | None = None,
    *,
    category: Literal["retrieval", "transformation"] = "transformation",
) -> Callable[..., object]:
    """Register one fully typed, documented, top-level trusted operation."""
    if function is None:
        return lambda decorated: code(decorated, category=category)
    if category not in {"retrieval", "transformation"}:
        raise ValueError(f"Unsupported @code operation category: {category!r}")
    if "<locals>" in function.__qualname__ or "." in function.__qualname__:
        raise TypeError("@code functions must be defined at module scope")
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    if "return" not in hints or hints["return"] in {Any, None, type(None)}:
        raise TypeError(f"{function.__name__} must declare one concrete return type")
    documentation = inspect.getdoc(function) or ""
    summary, arguments, result = _parse_docstring(documentation)
    inputs: list[tuple[str, str, str]] = []
    defaults: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise TypeError(f"{function.__name__} has unsupported parameter {parameter.name!r}")
        annotation = hints.get(parameter.name, inspect.Signature.empty)
        if annotation in {inspect.Signature.empty, Any}:
            raise TypeError(f"{function.__name__}.{parameter.name} must have a concrete type")
        if parameter.name not in arguments:
            raise TypeError(f"Docstring Args section is missing {parameter.name!r}")
        inputs.append(
            (parameter.name, _type_name(annotation), arguments[parameter.name])
        )
        if parameter.default is not inspect.Signature.empty:
            defaults[parameter.name] = parameter.default
    operation_id = function.__name__
    if operation_id in _OPERATIONS:
        existing = _OPERATIONS[operation_id]
        raise ValueError(
            f"Duplicate @code function name {operation_id!r}: "
            f"{existing.qualified_id!r} and "
            f"{f'{function.__module__}.{function.__qualname__}'!r}. "
            "@code function names must be globally unique."
        )
    _OPERATIONS[operation_id] = RegisteredOperation(
        id=operation_id,
        function=function,
        module=function.__module__,
        name=function.__name__,
        description=summary,
        inputs=tuple(inputs),
        defaults=defaults,
        output_type=_type_name(hints["return"]),
        output_description=result,
        category=category,
    )
    facade = sys.modules.get("llm_geo.ops")
    if facade is not None:
        setattr(facade, function.__name__, function)
    return function


def operation_aliases(operation: RegisteredOperation) -> tuple[str, ...]:
    """Return canonical and legacy identifiers accepted at system boundaries."""
    return operation.id, operation.qualified_id


def registered_operations() -> tuple[RegisteredOperation, ...]:
    """Return registered operations in a stable order for graph construction."""
    return tuple(_OPERATIONS[operation_id] for operation_id in sorted(_OPERATIONS))


def _parse_docstring(documentation: str) -> tuple[str, dict[str, str], str]:
    lines = [line.rstrip() for line in documentation.splitlines()]
    summary = next((line.strip() for line in lines if line.strip()), "")
    if not summary:
        raise TypeError("@code functions require a summary docstring")
    try:
        args_index = next(index for index, line in enumerate(lines) if line.strip() == "Args:")
        returns_index = next(index for index, line in enumerate(lines) if line.strip() == "Returns:")
    except StopIteration as error:
        raise TypeError("@code docstrings require Args: and Returns: sections") from error
    if returns_index <= args_index:
        raise TypeError("Args: must appear before Returns: in an @code docstring")
    arguments: dict[str, str] = {}
    for line in lines[args_index + 1 : returns_index]:
        stripped = line.strip()
        if not stripped:
            continue
        name, separator, description = stripped.partition(":")
        if not separator or not name or not description.strip():
            raise TypeError("Each Args entry must use 'name: description'")
        arguments[name.strip()] = description.strip()
    result = next(
        (line.strip() for line in lines[returns_index + 1 :] if line.strip()), ""
    )
    if not result:
        raise TypeError("@code docstrings require a Returns description")
    return summary, arguments, result


def _type_name(annotation: object) -> str:
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))
