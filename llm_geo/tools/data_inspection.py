"""Local, lossless GeoJSON source inspection."""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
from langchain_core.tools import tool
from toon_format import decode, encode

from llm_geo.middleware.logging import get_logger
from llm_geo.utils.models import DataSource


def to_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def to_toon(data: Any) -> str:
    """Encode JSON-compatible data compactly for an LLM prompt or tool message."""
    return encode(json.loads(to_json(data)))


def from_toon(text: str) -> Any:
    """Decode a TOON tool message into ordinary Python data."""
    return decode(text)


@tool
def inspect_vector(source: str, sample_rows: int = 5) -> str:
    """Inspect a GeoJSON dataset and return spatial metadata plus sample attributes."""
    if not 1 <= sample_rows <= 20:
        raise ValueError("sample_rows must be between 1 and 20")
    frame = gpd.read_file(source)
    attributes = frame.drop(columns=frame.geometry.name)
    return to_toon(
        {
            "source": source,
            "feature_count": len(frame),
            "crs": frame.crs,
            "geometry_types": sorted(frame.geom_type.dropna().unique().tolist()),
            "bounds": frame.total_bounds.tolist() if not frame.empty else None,
            "columns": list(attributes.columns),
            "dtypes": {
                column: str(dtype) for column, dtype in attributes.dtypes.items()
            },
            "sample": attributes.head(sample_rows)
            .where(attributes.notna(), None)
            .to_dict(orient="records"),
            "null_geometry_count": int(frame.geometry.isna().sum()),
            "invalid_geometry_count": int(
                (~frame.geometry.is_valid & frame.geometry.notna()).sum()
            ),
        }
    )


def inspect_source(source: DataSource) -> DataSource:
    """Inspect a provider GeoJSON source and retain errors as planner-visible metadata."""
    try:
        metadata = from_toon(inspect_vector.invoke({"source": source.location}))
        get_logger().info(
            "Data inspected | format=%s | source=%s", source.format, source.location
        )
        return source.model_copy(update={"metadata": metadata, "inspection_error": None})
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        get_logger().warning(
            "Data inspection unavailable | source=%s | reason=%s",
            source.location,
            message,
        )
        return source.model_copy(update={"inspection_error": message})
