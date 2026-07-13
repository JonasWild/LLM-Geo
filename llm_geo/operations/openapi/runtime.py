"""Shared HTTP runtime for generated OpenAPI operations."""

from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import geopandas as gpd
import requests


class OpenAPIOperationError(RuntimeError):
    """An OpenAPI operation could not produce its declared JSON response."""


def _service_environment_name(service: str, suffix: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in service)
    return f"LLM_GEO_OPENAPI_{normalized.upper()}_{suffix}"


@lru_cache(maxsize=16)
def _session(service: str) -> requests.Session:
    return requests.Session()


def clear_session_cache() -> None:
    """Clear cached HTTP sessions after configuration changes or in tests."""
    _session.cache_clear()


def invoke_json(
    *,
    service: str,
    method: str,
    path: str,
    default_base_url: str,
    path_parameters: dict[str, object] | None = None,
    query_parameters: dict[str, object] | None = None,
    header_parameters: dict[str, object] | None = None,
    json_body: object | None = None,
    api_key_environment: str | None = None,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
) -> Any:
    """Invoke one endpoint and return its decoded JSON response."""
    base_url = os.getenv(
        _service_environment_name(service, "URL"), default_base_url
    ).rstrip("/")
    timeout_raw = os.getenv("LLM_GEO_OPENAPI_TIMEOUT", "30")
    try:
        timeout = float(timeout_raw)
    except ValueError as error:
        raise ValueError("LLM_GEO_OPENAPI_TIMEOUT must be numeric") from error
    rendered_path = path
    for name, value in (path_parameters or {}).items():
        rendered_path = rendered_path.replace(
            "{" + name + "}", quote(str(value), safe="")
        )
    if "{" in rendered_path or "}" in rendered_path:
        raise OpenAPIOperationError(f"Unresolved path parameter in {rendered_path!r}")
    headers = {
        name: str(value)
        for name, value in (header_parameters or {}).items()
        if value is not None
    }
    key_environment = api_key_environment or _service_environment_name(service, "API_KEY")
    api_key = os.getenv(key_environment)
    if api_key:
        headers[auth_header] = f"{auth_scheme} {api_key}".strip()
    try:
        params = {
                name: value
                for name, value in (query_parameters or {}).items()
                if value is not None
            }
        response = _session(service).request(
            method=method,
            url=f"{base_url}{rendered_path}",
            params=params,
            headers=headers,
            json=json_body,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise OpenAPIOperationError(
            f"{service} {method.upper()} {path} failed: {error}.\nParams: {params}\nBody: {json_body}"
        ) from error
    try:
        return response.json()
    except ValueError as error:
        raise OpenAPIOperationError(
            f"{service} {method.upper()} {path} returned invalid JSON"
        ) from error


def geojson_to_geodataframe(payload: Any, *, source: str) -> gpd.GeoDataFrame:
    """Convert a decoded GeoJSON JSON payload (FeatureCollection, Feature, or bare Geometry)
    into a GeoDataFrame with provenance metadata in `.attrs["provenance"]`."""
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
    elif isinstance(payload, dict) and payload.get("type") == "Feature":
        features = [payload]
    elif isinstance(payload, dict) and "coordinates" in payload:
        features = [{"type": "Feature", "properties": {}, "geometry": payload}]
    else:
        raise OpenAPIOperationError(f"{source} did not return a recognizable GeoJSON payload")
    gdf = (
        gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        if features else gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    )
    gdf.attrs["provenance"] = {
        "source": source, "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return gdf
