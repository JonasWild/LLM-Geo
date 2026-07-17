"""Compile a node's generated code, enforce its typed `run` signature against the node spec,
and validate input/output values at the node boundary.

Validation is layered:
- coarse: every port value must match its `PortSpec.type` (plus GeoDataFrame integrity and
  provenance rules) -- enforced at contract-test time AND at real execution time.
- fine: the coder self-declares a precise contract as the `run` return annotation (a TypedDict);
  the contract test holds the output to it via pydantic, with pydantic's error paths as feedback.
"""
from __future__ import annotations

import inspect
import traceback
from typing import Any, Mapping, get_origin, get_type_hints

import geopandas as gpd
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import is_typeddict

from .models import ContractResult, NodeKind, NodeSpec
from .synthetic import make_inputs

ANNOTATION_BY_PORT_TYPE = {
    "str": "str", "int": "int", "float": "float", "bool": "bool", "dict": "dict",
    "GeoDataFrame": "gpd.GeoDataFrame",
}


def coarse_type_ok(value: Any, type_name: str) -> bool:
    match type_name:
        case "bool":
            return isinstance(value, bool)
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "str":
            return isinstance(value, str)
        case "dict":
            return isinstance(value, Mapping)
        case "GeoDataFrame":
            return isinstance(value, gpd.GeoDataFrame)
    return False


def field_type_ok(value: Any, type_name: str) -> bool:
    if type_name.startswith("list["):
        inner = type_name[5:-1]
        return isinstance(value, list) and all(field_type_ok(item, inner) for item in value)
    return coarse_type_ok(value, type_name)


def dict_field_errors(label: str, port: Any, value: Mapping) -> list[str]:
    """Check a dict value against the port's declared per-key field contract: every declared
    key present with its type, and no undeclared keys (which usually signal an accidental
    extra wrapper level around the value)."""
    if not port.fields:
        return []
    errors = []
    for name, field in port.fields.items():
        if name not in value:
            errors.append(f"{label}: missing declared key '{name}'")
        elif not field_type_ok(value[name], field.type):
            errors.append(
                f"{label}: key '{name}' must be {field.type}, got {type(value[name]).__name__}"
            )
    for name in value:
        if name not in port.fields:
            errors.append(
                f"{label}: undeclared key '{name}' -- the value must have exactly the keys "
                f"{sorted(port.fields)}, with no extra wrapper level"
            )
    return errors


def _frame_errors(label: str, frame: gpd.GeoDataFrame) -> list[str]:
    errors = []
    try:
        frame.geometry
    except Exception:
        return [f"{label}: GeoDataFrame has no active geometry column (pass geometry=... or call .set_geometry)"]
    if frame.crs is None:
        errors.append(f"{label}: GeoDataFrame has no CRS set")
    return errors


def _hint_matches(hint: Any, type_name: str) -> bool:
    origin = get_origin(hint) or hint
    match type_name:
        case "GeoDataFrame":
            return getattr(origin, "__name__", "") == "GeoDataFrame"
        case "dict":
            return origin in (dict, Mapping) or is_typeddict(hint)
        case "str":
            return origin is str
        case "bool":
            return origin is bool
        case "int":
            return origin is int
        case "float":
            return origin in (float, int)
    return False


def signature_errors(node: NodeSpec, fn: Any) -> list[str]:
    """The generated `run` must take exactly the node's inputs+params as named annotated
    parameters (no *args/**kwargs) and declare a return annotation."""
    sig = inspect.signature(fn)
    if any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in sig.parameters.values()
    ):
        return ["run() must not use *args/**kwargs: declare every input and param as a named parameter"]
    errors = []
    expected = set(node.inputs) | set(node.params)
    got = set(sig.parameters)
    if missing := sorted(expected - got):
        errors.append(f"run() is missing parameters {missing}")
    if extra := sorted(got - expected):
        errors.append(f"run() has parameters {extra} that are neither declared inputs nor params")
    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        return errors + [f"could not resolve type hints on run(): {exc}"]
    for name, port in node.inputs.items():
        if name not in sig.parameters:
            continue
        hint = hints.get(name)
        if hint is None:
            errors.append(f"parameter '{name}' must be annotated as {ANNOTATION_BY_PORT_TYPE[port.type]}")
        elif not _hint_matches(hint, port.type):
            errors.append(
                f"parameter '{name}' is annotated {hint!r} but the node spec declares it as {port.type}"
            )
    if "return" not in hints:
        errors.append("run() must declare a return annotation: a TypedDict of the declared outputs (preferred) or dict")
    return errors


def input_errors(node: NodeSpec, resolved: Mapping[str, Any]) -> list[str]:
    """Coarse-check resolved input values before calling the node."""
    errors = []
    for name, port in node.inputs.items():
        if name not in resolved:
            errors.append(f"input '{name}' was not fed by any dependency output or param")
        elif not coarse_type_ok(resolved[name], port.type):
            errors.append(
                f"input '{name}' must be {port.type}, got {type(resolved[name]).__name__}"
            )
        elif port.type == "GeoDataFrame":
            errors += _frame_errors(f"input '{name}'", resolved[name])
        elif port.type == "dict":
            errors += dict_field_errors(f"input '{name}'", port, resolved[name])
    return errors


def output_errors(node: NodeSpec, output: Any) -> list[str]:
    """Coarse-check a node's returned dict against its declared output ports."""
    if not isinstance(output, Mapping):
        return [f"run() must return a dict, got {type(output).__name__}"]
    errors = []
    for name, port in node.outputs.items():
        if name not in output:
            errors.append(f"missing declared output '{name}'")
            continue
        value = output[name]
        if not coarse_type_ok(value, port.type):
            errors.append(f"output '{name}' must be {port.type}, got {type(value).__name__}")
            continue
        if port.type == "GeoDataFrame":
            errors += _frame_errors(f"output '{name}'", value)
            if node.kind is NodeKind.retrieval and "provenance" not in getattr(value, "attrs", {}):
                errors.append(f"retrieval output '{name}' is missing .attrs['provenance'] metadata")
        elif port.type == "dict":
            errors += dict_field_errors(f"output '{name}'", port, value)
    return errors


def fine_output_validator(fn: Any):
    """Build a pydantic validator from run()'s self-declared return TypedDict, if usable.

    Returns None when the return annotation is a plain dict or cannot be adapted (e.g.
    `typing.TypedDict` on Python 3.11, which pydantic rejects in favor of typing_extensions).
    """
    try:
        return_hint = get_type_hints(fn).get("return")
        if return_hint is None or not is_typeddict(return_hint):
            return None
        # pydantic takes a TypedDict's config from the class itself; the annotation may name
        # arbitrary types like GeoDataFrame, which then validate by isinstance.
        if not hasattr(return_hint, "__pydantic_config__"):
            return_hint.__pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)
        adapter = TypeAdapter(return_hint)
        # A TypedDict defined in exec'd node code defers the initial schema build (pydantic
        # cannot see the exec namespace from our frame); an explicit rebuild resolves it.
        adapter.rebuild(raise_errors=True)
    except Exception:
        return None

    def validate(output: Mapping[str, Any]) -> None:
        adapter.validate_python(dict(output))

    return validate


def compile_node(code: str, node: NodeSpec | None = None):
    ns: dict = {}
    exec(compile(code, "<node>", "exec"), ns)
    fn = ns.get("run")
    if not callable(fn):
        raise ValueError("code must define a callable `run(...)` function")
    if node is not None and (errors := signature_errors(node, fn)):
        raise ValueError("run() signature does not match the node spec:\n- " + "\n- ".join(errors))
    return fn


def run_contract(node: NodeSpec, code: str) -> ContractResult:
    try:
        fn = compile_node(code, node)
        # Params are passed with their real literal values, exactly as the executor will;
        # when a name is both input and param, the known literal beats a synthetic stand-in.
        output = fn(**{**make_inputs(node.inputs), **node.params})
        if errors := output_errors(node, output):
            raise TypeError("output contract violated:\n- " + "\n- ".join(errors))
        if validate := fine_output_validator(fn):
            validate(output)  # raises pydantic.ValidationError with precise field paths
        return ContractResult(ok=True, output=output)
    except Exception:
        return ContractResult(ok=False, error=traceback.format_exc(limit=4))
