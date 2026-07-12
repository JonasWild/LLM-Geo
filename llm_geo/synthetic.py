"""Synthetic input generation so a node's contract can be tested without running upstream nodes."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import Point


def _sample_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0.0, 0.0), Point(1.0, 1.0)], crs="EPSG:4326",
    )


def make_value(type_name: str) -> Any:
    match type_name:
        case "GeoDataFrame":
            return _sample_gdf()
        case "float" | "int":
            return 1.5
        case "bool":
            return True
        case "dict":
            return {"synthetic": True}
        case _:
            return "synthetic-string"


def make_inputs(inputs: dict[str, str]) -> dict[str, Any]:
    return {name: make_value(type_name) for name, type_name in inputs.items()}
