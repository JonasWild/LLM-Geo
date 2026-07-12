"""Compile a node's generated code and run it against synthetic inputs to check its contract."""
from __future__ import annotations

import traceback

import geopandas as gpd

from .models import ContractResult, NodeSpec
from .synthetic import make_inputs

_TYPE_CHECK = {
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "float": lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
    "int": lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
    "dict": lambda v: isinstance(v, dict),
    "GeoDataFrame": lambda v: isinstance(v, gpd.GeoDataFrame),
}


def _validate_output(node: NodeSpec, output: dict) -> None:
    if not isinstance(output, dict):
        raise TypeError(f"run() must return a dict, got {type(output).__name__}")
    for name, type_name in node.outputs.items():
        if name not in output:
            raise ValueError(f"missing declared output '{name}'")
        value = output[name]
        check = _TYPE_CHECK.get(type_name)
        if check is not None and not check(value):
            raise TypeError(f"output '{name}' must be a {type_name}, got {type(value).__name__}")
        if type_name == "GeoDataFrame" and node.kind.value == "retrieval" and "provenance" not in value.attrs:
            raise ValueError(f"retrieval output '{name}' is missing provenance metadata in .attrs['provenance']")


def compile_node(code: str):
    ns: dict = {}
    exec(compile(code, "<node>", "exec"), ns)
    fn = ns.get("run")
    if not callable(fn):
        raise ValueError("code must define a callable `run(**inputs) -> dict`")
    return fn


def run_contract(node: NodeSpec, code: str) -> ContractResult:
    try:
        fn = compile_node(code)
        output = fn(**make_inputs(node.inputs))
        _validate_output(node, output)
        return ContractResult(ok=True, output=output)
    except Exception:
        return ContractResult(ok=False, error=traceback.format_exc(limit=4))
