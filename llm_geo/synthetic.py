"""Synthetic input generation so a node's contract can be tested without running upstream nodes."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from .models import PortSpec

_GEOMS = {
    "Point": [Point(0.0, 0.0), Point(1.0, 1.0)],
    "LineString": [LineString([(0, 0), (1, 1)]), LineString([(1, 0), (0, 1)])],
    "Polygon": [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 2), (3, 2), (3, 3)])],
}

_COLUMN_VALUES = {"int": [1, 2], "float": [0.5, 1.5], "bool": [True, False]}

# Strict per-type check for planner-provided examples (bool is not an acceptable int/float).
_EXAMPLE_OK = {
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "dict": lambda v: isinstance(v, dict),
}


def _sample_gdf(port: PortSpec) -> gpd.GeoDataFrame:
    geometry = _GEOMS.get(port.geometry or "Point", _GEOMS["Point"])
    columns = port.columns or {"name": "str"}
    data = {name: _COLUMN_VALUES.get(dtype, ["a", "b"]) for name, dtype in columns.items()}
    return gpd.GeoDataFrame(data, geometry=geometry, crs=port.crs or "EPSG:4326")


def make_value(port: PortSpec | str) -> Any:
    if isinstance(port, str):
        port = PortSpec(type=port)
    if port.example is not None and _EXAMPLE_OK.get(port.type, lambda _: False)(port.example):
        return port.example
    match port.type:
        case "GeoDataFrame":
            return _sample_gdf(port)
        case "float":
            return 1.5
        case "int":
            return 2
        case "bool":
            return True
        case "dict":
            return {"synthetic": True}
        case _:
            return "synthetic-string"


def make_inputs(inputs: dict[str, PortSpec | str]) -> dict[str, Any]:
    return {name: make_value(port) for name, port in inputs.items()}
