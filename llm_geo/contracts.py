"""Compile a node's generated code and run it against synthetic inputs to check its contract."""
from __future__ import annotations

import traceback
import typing
from typing import get_origin, get_args, Any, Union, Mapping, Sequence

import geopandas as gpd

from .models import ContractResult, NodeSpec
from .synthetic import make_inputs

DEBUG_TYPE_CHECK = True
def _dbg(msg: str) -> None:
    if DEBUG_TYPE_CHECK: print(f"[type-check] {msg}")

_TYPE_NS = {
    **typing.__dict__,
    "str": str, "int": int, "float": float, "bool": bool, "dict": dict,
    "list": list, "tuple": tuple, "set": set, "Any": Any,
    # GeoDataFrame as string alias (duck-typed later)
    "GeoDataFrame": "GeoDataFrame",
}

def _parse_type(t: str | Any) -> Any:
    """Convert 'list[tuple[float, float]]' -> list[tuple[float, float]] (GenericAlias)."""
    if not isinstance(t, str):
        return t
    # _eval_type handles ForwardRef, unions, generics (Python 3.9+)
    return eval(t, _TYPE_NS, _TYPE_NS)


# ---- 2. Primitive / duck checks ----
def _check_primitive(val: Any, typ: Any) -> bool:
    if typ is bool:   return isinstance(val, bool)
    if typ is int:    return isinstance(val, int) and not isinstance(val, bool)
    if typ is float:  return isinstance(val, (int, float)) and not isinstance(val, bool)
    if typ is str:    return isinstance(val, str)
    if typ is dict:   return isinstance(val, Mapping)
    if typ is list:   return isinstance(val, Sequence) and not isinstance(val, (str, bytes))
    if typ is tuple:  return isinstance(val, tuple)
    if typ is Any:    return True
    if isinstance(typ, str): return type(val).__name__ == typ  # "GeoDataFrame"
    return isinstance(val, typ)


# ---- 3. Recursive validator ----
def check_type(val: Any, typ: Any, _path: str = "$") -> bool:
    origin, args = get_origin(typ), get_args(typ)
    _dbg(f"{_path}: checking {type(val).__name__} vs {typ}")

    if origin is Union:
        ok = any(check_type(val, a, _path) for a in args)
        if not ok: _dbg(f"{_path}: FAIL union {typ}")
        return ok

    if origin:
        if not _check_primitive(val, origin):
            _dbg(f"{_path}: FAIL container origin {origin}")
            return False
        if not args: return True

        if origin is tuple:
            if len(args) == 2 and args[1] is ...:
                return all(check_type(v, args[0], f"{_path}[{i}]") for i, v in enumerate(val))
            return len(val) == len(args) and all(check_type(v, a, f"{_path}[{i}]") for i, (v, a) in enumerate(zip(val, args)))

        if origin is dict:
            k_t, v_t = args
            return all(check_type(k, k_t, f"{_path}.key") and check_type(v, v_t, f"{_path}[{k}]") for k, v in val.items())

        elem_t = args[0]
        return all(check_type(v, elem_t, f"{_path}[{i}]") for i, v in enumerate(val))

    ok = _check_primitive(val, typ)
    if not ok: _dbg(f"{_path}: FAIL primitive {typ}")
    return ok


# ---- 4. Validator against NodeSpec ----
def _validate_output(node_spec: "NodeSpec", output: dict) -> None:
    if not isinstance(output, dict):
        raise TypeError(f"run() must return dict, got {type(output).__name__}")

    # Parse spec strings ONCE
    type_map = {k: _parse_type(v) for k, v in node_spec.outputs.items()}

    kind = node_spec.kind.value

    for name, typ in type_map.items():
        if name not in output:
            raise ValueError(f"missing declared output '{name}'")

        val = output[name]
        if not check_type(val, typ, f"$.{name}"):
            _dbg(f"$.{name}: FINAL FAIL → {type(val).__name__} vs {typ}")
            raise TypeError(f"output '{name}' must be {typ}, got {type(val).__name__}")

        # Provenance hook (uses string name from spec)
        if kind == "retrieval" and node_spec.outputs.get(name) == "GeoDataFrame":
            if "provenance" not in getattr(val, "attrs", {}):
                raise ValueError("retrieval GeoDataFrame missing .attrs['provenance']")


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
