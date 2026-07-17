"""Compile a node's generated code, enforce its typed signature, and run it against synthetic
inputs to check its contract. `validate_value` is also used by the executor at every DAG edge."""
from __future__ import annotations

import inspect
import traceback

import geopandas as gpd

from .models import ContractResult, NodeSpec, PortSpec
from .synthetic import make_inputs

_TYPE_CHECK = {
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "float": lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
    "int": lambda v: not isinstance(v, bool) and isinstance(v, int),
    "dict": lambda v: isinstance(v, dict),
    "GeoDataFrame": lambda v: isinstance(v, gpd.GeoDataFrame),
}


def validate_value(name: str, port: PortSpec, value: object) -> None:
    """Raise with a precise message if `value` violates the port's type or declared constraints."""
    if not _TYPE_CHECK[port.type](value):
        raise TypeError(f"{name} must be a {port.type}, got {type(value).__name__}")
    if port.type != "GeoDataFrame":
        return
    try:
        value.geometry
    except Exception:
        raise TypeError(f"{name}: GeoDataFrame has no active geometry column (use set_geometry)") from None
    if value.crs is None:
        raise ValueError(f"{name}: GeoDataFrame has no CRS set")
    if port.crs is not None and value.crs != port.crs:
        raise ValueError(f"{name}: expected CRS {port.crs}, got {value.crs}")
    if port.geometry not in (None, "any") and len(value):
        allowed = {port.geometry, f"Multi{port.geometry}"}
        bad = set(value.geom_type.unique()) - allowed
        if bad:
            raise ValueError(f"{name}: expected {port.geometry} geometry, got {sorted(bad)}")
    if port.columns:
        missing = [c for c in port.columns if c not in value.columns]
        if missing:
            raise ValueError(f"{name}: GeoDataFrame is missing declared columns {missing}")


def check_signature(fn, node: NodeSpec) -> None:
    """run() must declare exactly the node's inputs+params as explicit parameters -- no **kwargs."""
    parameters = inspect.signature(fn).parameters
    for p in parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"run() must declare explicit named parameters, not '*{p.name}'/'**{p.name}'")
    expected = set(node.inputs) | set(node.params)
    if set(parameters) != expected:
        raise TypeError(f"run() parameters {sorted(parameters)} must be exactly {sorted(expected)}")


def _validate_output(node: NodeSpec, output: dict) -> None:
    if not isinstance(output, dict):
        raise TypeError(f"run() must return a dict, got {type(output).__name__}")
    for name, port in node.outputs.items():
        if name not in output:
            raise ValueError(f"missing declared output '{name}'")
        validate_value(f"output '{name}'", port, output[name])
        if port.type == "GeoDataFrame" and node.kind.value == "retrieval" and "provenance" not in output[name].attrs:
            raise ValueError(f"retrieval output '{name}' is missing provenance metadata in .attrs['provenance']")


def compile_node(code: str):
    ns: dict = {"gpd": gpd}  # the generated signature annotates GeoDataFrame ports as gpd.GeoDataFrame
    exec(compile(code, "<node>", "exec"), ns)
    fn = ns.get("run")
    if not callable(fn):
        raise ValueError("code must define a callable `run(...) -> dict`")
    return fn


def run_contract(node: NodeSpec, code: str) -> ContractResult:
    try:
        fn = compile_node(code)
        check_signature(fn, node)
        output = fn(**{**make_inputs(node.inputs), **node.params})
        _validate_output(node, output)
        return ContractResult(ok=True, output=output)
    except Exception:
        return ContractResult(ok=False, error=traceback.format_exc(limit=4))
