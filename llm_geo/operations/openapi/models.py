"""Normalized contracts shared by the OpenAPI parser and renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ParameterLocation = Literal["path", "query", "header", "body"]


@dataclass(frozen=True)
class ParameterDefinition:
    """One Python argument and its corresponding HTTP input."""

    python_name: str
    wire_name: str
    location: ParameterLocation
    annotation: str
    description: str
    required: bool
    default: Any = None
    has_default: bool = False


@dataclass(frozen=True)
class OpenAPIDefinition:
    """One supported OpenAPI operation ready to render as Python."""

    function_name: str
    operation_id: str
    method: str
    path: str
    summary: str
    parameters: tuple[ParameterDefinition, ...]
    return_annotation: str
    return_description: str


@dataclass(frozen=True)
class ParseResult:
    """Supported definitions and explanations for skipped operations."""

    operations: tuple[OpenAPIDefinition, ...]
    skipped: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class GenerationResult:
    """Files and operation IDs produced by one successful generation."""

    module_path: Path
    spec_path: Path
    manifest_path: Path
    operation_ids: tuple[str, ...]
    changed: bool
