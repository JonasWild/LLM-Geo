"""Provider-tool contracts and validation for system-retrieved GeoJSON data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from llm_geo.tools.data_inspection import from_toon
from llm_geo.utils.models import DataSource


RetrievalTool = BaseTool


def provider_tool_instructions(destination: Path) -> str:
    """Return the provider contract supplied to the retrieval agent."""
    return f"""
Each provider tool must write one or more GeoJSON FeatureCollection files below:
{destination}

Each tool result is TOON with either one source object or a `sources` list.
Every source object requires `description`, `location`, `provider`, and optional
`request`. Set `location` to the created absolute local file path. Do not return URLs.
""".strip()


def validate_provider_results(
    raw_results: list[str], data_directory: Path
) -> list[DataSource]:
    """Accept only provider-produced, run-scoped GeoJSON FeatureCollections."""
    sources: list[DataSource] = []
    for raw_result in raw_results:
        try:
            payload = from_toon(raw_result)
        except ValueError as error:
            raise ValueError(f"Provider returned invalid TOON: {error}") from error
        entries = payload.get("sources") if isinstance(payload, dict) else None
        if entries is None:
            entries = [payload]
        if not isinstance(entries, list):
            raise ValueError("Provider result field 'sources' must be a list.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each provider source must be a JSON object.")
            source = DataSource.model_validate({**entry, "format": "GeoJSON"})
            path = _validate_geojson_file(source.location, data_directory)
            sources.append(source.model_copy(update={"location": str(path)}))
    if not sources:
        raise ValueError("No provider returned a GeoJSON source.")
    locations = [source.location for source in sources]
    if len(locations) != len(set(locations)):
        raise ValueError("Provider results contain duplicate GeoJSON locations.")
    return sources


def _validate_geojson_file(location: str, data_directory: Path) -> Path:
    path = Path(location).resolve()
    root = data_directory.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Provider file is outside the run data directory: {location}")
    if path.suffix.lower() not in {".geojson", ".json"}:
        raise ValueError(f"Provider file is not a GeoJSON path: {location}")
    if not path.is_file():
        raise ValueError(f"Provider file does not exist: {location}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Provider file is not valid JSON: {location}") from error
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError(f"Provider file is not a GeoJSON FeatureCollection: {location}")
    return path