"""Convert an OpenAPI document into small, renderer-independent definitions."""

from __future__ import annotations

import keyword
import re
from typing import Any

from llm_geo.operations.openapi.models import (
    OpenAPIDefinition,
    ParameterDefinition,
    ParseResult,
)

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "options", "head")


class UnsupportedOperation(ValueError):
    """Raised when an endpoint cannot be represented safely as an operation."""


def python_identifier(value: str, fallback: str = "value") -> str:
    """Return a stable snake_case Python identifier."""
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower() or fallback
    if value[0].isdigit():
        value = f"{fallback}_{value}"
    if keyword.iskeyword(value):
        value = f"{value}_"
    return value


def normalize_operation_ids(spec: dict[str, Any]) -> None:
    """Assign unique Python-compatible operation IDs in-place."""
    used: set[str] = set()
    for path, path_item in sorted(spec.get("paths", {}).items()):
        if not isinstance(path_item, dict):
            continue
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            fallback = python_identifier(f"{method}_{path}", "endpoint")
            base = python_identifier(str(operation.get("operationId") or fallback), "endpoint")
            name = base
            suffix = 2
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name)
            operation["operationId"] = name


def resolve_schema(schema: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a chain of local schema references."""
    seen: set[str] = set()
    current = schema
    while "$ref" in current:
        reference = current["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise UnsupportedOperation(f"external reference is unsupported: {reference!r}")
        if reference in seen:
            raise UnsupportedOperation(f"circular reference is unsupported: {reference}")
        seen.add(reference)
        value: Any = spec
        for part in reference[2:].split("/"):
            if not isinstance(value, dict) or part not in value:
                raise UnsupportedOperation(f"unresolved reference: {reference}")
            value = value[part]
        if not isinstance(value, dict):
            raise UnsupportedOperation(f"reference is not an object: {reference}")
        current = value
    return current


def schema_annotation(
    schema: dict[str, Any], spec: dict[str, Any], *, depth: int = 0
) -> str:
    """Map an OpenAPI schema to a concrete runtime-safe Python annotation."""
    if depth > 8:
        return "dict[str, Any]"
    schema = resolve_schema(schema, spec)
    nullable = bool(schema.get("nullable"))
    if "oneOf" in schema or "anyOf" in schema:
        variants = schema.get("oneOf") or schema.get("anyOf") or []
        annotations = [
            schema_annotation(item, spec, depth=depth + 1)
            for item in variants
            if isinstance(item, dict)
        ]
        annotation = " | ".join(dict.fromkeys(annotations)) or "object"
    elif "allOf" in schema:
        annotation = "dict[str, Any]"
    else:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            nullable = nullable or "null" in schema_type
            schema_type = next((item for item in schema_type if item != "null"), None)
        if schema_type == "string":
            annotation = "str"
        elif schema_type == "integer":
            annotation = "int"
        elif schema_type == "number":
            annotation = "float"
        elif schema_type == "boolean":
            annotation = "bool"
        elif schema_type == "array":
            items = schema.get("items", {})
            item_annotation = (
                schema_annotation(items, spec, depth=depth + 1)
                if isinstance(items, dict)
                else "object"
            )
            annotation = f"list[{item_annotation}]"
        elif schema_type in {"object", None} or "properties" in schema:
            annotation = "dict[str, Any]"
        elif schema_type == "null":
            annotation = "object"
            nullable = True
        else:
            annotation = "object"
    if nullable and "None" not in annotation.split(" | "):
        return f"{annotation} | None"
    return annotation


def _description(value: Any, fallback: str) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace('"""', "'''")
    return text.rstrip(".") + "." if text else fallback


def _unique_name(wire_name: str, location: str, used: set[str]) -> str:
    base = python_identifier(wire_name)
    candidate = base if base not in used else python_identifier(f"{location}_{base}")
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _parameter_definition(
    raw: dict[str, Any], spec: dict[str, Any], used: set[str]
) -> ParameterDefinition:
    location = raw.get("in")
    if location not in {"path", "query", "header"}:
        raise UnsupportedOperation(f"parameter location {location!r} is unsupported")
    wire_name = str(raw.get("name") or "value")
    schema = raw.get("schema", {})
    if not isinstance(schema, dict):
        schema = {}
    required = bool(raw.get("required")) or location == "path"
    has_schema_default = "default" in schema
    annotation = schema_annotation(schema, spec)
    if not required and not has_schema_default and "None" not in annotation.split(" | "):
        annotation = f"{annotation} | None"
    return ParameterDefinition(
        python_name=_unique_name(wire_name, location, used),
        wire_name=wire_name,
        location=location,
        annotation=annotation,
        description=_description(
            raw.get("description"), f"{location.title()} parameter {wire_name}."
        ),
        required=required,
        default=schema.get("default"),
        has_default=has_schema_default,
    )


def _body_parameters(
    request_body: dict[str, Any], spec: dict[str, Any], used: set[str]
) -> list[ParameterDefinition]:
    request_body = resolve_schema(request_body, spec)
    content = request_body.get("content", {})
    media = content.get("application/json") if isinstance(content, dict) else None
    if not isinstance(media, dict) or not isinstance(media.get("schema"), dict):
        raise UnsupportedOperation("only application/json request bodies are supported")
    schema = resolve_schema(media["schema"], spec)
    body_required = bool(request_body.get("required"))
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        annotation = schema_annotation(schema, spec)
        if not body_required and "None" not in annotation.split(" | "):
            annotation = f"{annotation} | None"
        return [
            ParameterDefinition(
                python_name=_unique_name("body", "body", used),
                wire_name="body",
                location="body",
                annotation=annotation,
                description=_description(
                    request_body.get("description"), "JSON request body."
                ),
                required=body_required,
            )
        ]
    required_properties = set(schema.get("required", []))
    result: list[ParameterDefinition] = []
    for wire_name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            property_schema = {}
        required = body_required and wire_name in required_properties
        has_default = "default" in property_schema
        annotation = schema_annotation(property_schema, spec)
        if not required and not has_default and "None" not in annotation.split(" | "):
            annotation = f"{annotation} | None"
        result.append(
            ParameterDefinition(
                python_name=_unique_name(str(wire_name), "body", used),
                wire_name=str(wire_name),
                location="body",
                annotation=annotation,
                description=_description(
                    property_schema.get("description"), f"JSON field {wire_name}."
                ),
                required=required,
                default=property_schema.get("default"),
                has_default=has_default,
            )
        )
    return result


def _response_contract(
    operation: dict[str, Any], spec: dict[str, Any]
) -> tuple[str, str]:
    responses = operation.get("responses", {})
    if not isinstance(responses, dict):
        raise UnsupportedOperation("responses must be an object")
    successful = sorted(
        (str(code), response)
        for code, response in responses.items()
        if (str(code).startswith("2") or str(code) == "default")
        and isinstance(response, dict)
    )
    for _, response in successful:
        response = resolve_schema(response, spec)
        content = response.get("content", {})
        media = content.get("application/json") if isinstance(content, dict) else None
        schema = media.get("schema") if isinstance(media, dict) else None
        if isinstance(schema, dict):
            return (
                schema_annotation(schema, spec),
                _description(response.get("description"), "Decoded JSON response."),
            )
    raise UnsupportedOperation("no successful application/json response schema")


def parse_openapi(spec: dict[str, Any]) -> ParseResult:
    """Parse all safely representable operations in an OpenAPI document."""
    if not isinstance(spec.get("paths"), dict):
        raise ValueError("OpenAPI document must contain a paths object")
    normalize_operation_ids(spec)
    definitions: list[OpenAPIDefinition] = []
    skipped: list[dict[str, str]] = []
    for path, path_item in sorted(spec["paths"].items()):
        if not isinstance(path_item, dict):
            continue
        shared_parameters = path_item.get("parameters", [])
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            operation_id = str(operation["operationId"])
            try:
                if operation.get("callbacks"):
                    raise UnsupportedOperation("callback operations are unsupported")
                used: set[str] = set()
                parameters: list[ParameterDefinition] = []
                combined_parameters: dict[tuple[str, str], dict[str, Any]] = {}
                raw_parameters = shared_parameters if isinstance(shared_parameters, list) else []
                operation_parameters = operation.get("parameters", [])
                for raw in [
                    *raw_parameters,
                    *(operation_parameters if isinstance(operation_parameters, list) else []),
                ]:
                    if not isinstance(raw, dict):
                        continue
                    raw = resolve_schema(raw, spec)
                    key = (str(raw.get("name")), str(raw.get("in")))
                    combined_parameters[key] = raw
                for raw in combined_parameters.values():
                    parameters.append(_parameter_definition(raw, spec, used))
                request_body = operation.get("requestBody")
                if isinstance(request_body, dict):
                    parameters.extend(_body_parameters(request_body, spec, used))
                parameters.sort(
                    key=lambda item: item.has_default or not item.required
                )
                return_annotation, return_description = _response_contract(operation, spec)
                description = _description(
                    operation.get("description") or operation.get("summary"),
                    f"Call the {operation_id} endpoint.",
                )
                definitions.append(
                    OpenAPIDefinition(
                        function_name=operation_id,
                        operation_id=operation_id,
                        method=method.upper(),
                        path=str(path),
                        description=description,
                        parameters=tuple(parameters),
                        return_annotation=return_annotation,
                        return_description=return_description,
                    )
                )
            except UnsupportedOperation as error:
                skipped.append(
                    {
                        "operation_id": operation_id,
                        "method": method.upper(),
                        "path": str(path),
                        "reason": str(error),
                    }
                )
    return ParseResult(tuple(definitions), tuple(skipped))
