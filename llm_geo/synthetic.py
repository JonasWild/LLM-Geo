"""Synthetic input generation so a node's contract can be tested without running upstream nodes."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import geopandas as gpd
from shapely.geometry import Point

from .models import PortSpec


def _sample_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0.0, 0.0), Point(1.0, 1.0)], crs="EPSG:4326",
    )


def field_value(type_name: str) -> Any:
    if type_name.startswith("list["):
        inner = type_name[5:-1]
        return [field_value(inner), field_value(inner)]
    match type_name:
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


def make_value(port: PortSpec) -> Any:
    # PortSpec guarantees a present example matches the declared type AND fields, so prefer it:
    # the contract test then runs against the planner's intended realistic value.
    if port.example is not None:
        return float(port.example) if port.type == "float" else deepcopy(port.example)
    if port.type == "GeoDataFrame":
        return _sample_gdf()
    if port.type == "dict" and port.fields:
        return {name: field_value(field.type) for name, field in port.fields.items()}
    return field_value(port.type)


def make_inputs(inputs: dict[str, PortSpec]) -> dict[str, Any]:
    return {name: make_value(port) for name, port in inputs.items()}
