"""Public OpenStreetMap retrieval tools that materialize GeoJSON datasets."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests
import geopandas as gpd

from llm_geo.middleware.logging import get_logger, http_logging_enabled
from llm_geo.operations import code


OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
)
OVERPASS_ENDPOINT = OVERPASS_ENDPOINTS[0]
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
DEFAULT_TIMEOUT_SECONDS = 90
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0
_last_nominatim_request = 0.0


def _write_feature_collection(output_path: str, collection: dict[str, Any]) -> Path:
    path = Path(output_path).resolve()
    if path.suffix.lower() not in {".geojson", ".json"}:
        raise ValueError("output_path must end in .geojson or .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    return path


def _as_wgs84_frame(collection: dict[str, Any]) -> gpd.GeoDataFrame:
    """Convert a provider FeatureCollection into the common runtime representation."""
    if not collection["features"]:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    frame = gpd.GeoDataFrame.from_features(collection["features"], crs="EPSG:4326")
    return frame.set_crs("EPSG:4326", allow_override=True)


def _overpass_feature(element: dict[str, Any]) -> dict[str, Any] | None:
    element_type = element.get("type")
    element_id = element.get("id")
    properties = {"osm_type": element_type, "osm_id": element_id, **element.get("tags", {})}
    if element_type == "node" and "lat" in element and "lon" in element:
        geometry = {"type": "Point", "coordinates": [element["lon"], element["lat"]]}
    elif element_type in {"way", "relation"} and element.get("geometry"):
        coordinates = [
            [point["lon"], point["lat"]]
            for point in element["geometry"]
            if "lat" in point and "lon" in point
        ]
        if len(coordinates) < 2:
            return None
        if coordinates[0] == coordinates[-1] and len(coordinates) >= 4:
            geometry = {"type": "Polygon", "coordinates": [coordinates]}
        else:
            geometry = {"type": "LineString", "coordinates": coordinates}
    else:
        return None
    return {"type": "Feature", "properties": properties, "geometry": geometry}


def _overpass_feature_collection(payload: dict[str, Any]) -> dict[str, Any]:
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Overpass response did not contain an elements list")
    features = [feature for element in elements if isinstance(element, dict) if (feature := _overpass_feature(element))]
    return {"type": "FeatureCollection", "features": features}


def _nominatim_user_agent() -> str:
    return os.getenv(
        "NOMINATIM_USER_AGENT",
        "LLM-GEO/0.2 (contact: jonas.wild2@hotmail.de)",
    ).strip()


def _overpass_user_agent() -> str:
    return os.getenv(
        "OVERPASS_USER_AGENT", "LLM-GEO/0.2 (contact: jonas.wild2@hotmail.de)"
    ).strip()


def _overpass_endpoints() -> tuple[str, ...]:
    """Return a configured endpoint or the default public mirror pool."""
    configured_endpoint = os.getenv("OVERPASS_URL", "").strip()
    return (configured_endpoint,) if configured_endpoint else OVERPASS_ENDPOINTS


def _nominatim_endpoint() -> str:
    """Return the configured Nominatim search endpoint."""
    return os.getenv("NOMINATIM_URL", "").strip() or NOMINATIM_ENDPOINT


def _send_request(
    *,
    provider: str,
    method: str,
    endpoint: str,
    send: Callable[[], requests.Response],
) -> requests.Response:
    """Send a provider request and log only safe request/response metadata."""
    log_http = http_logging_enabled()
    logger = get_logger()
    started_at = time.monotonic()
    if log_http:
        logger.info(
            "HTTP request | provider=%s | method=%s | endpoint=%s",
            provider,
            method,
            endpoint,
        )
    try:
        response = send()
    except requests.RequestException:
        if log_http:
            logger.exception(
                "HTTP request failed | provider=%s | method=%s | endpoint=%s | duration_seconds=%.3f",
                provider,
                method,
                endpoint,
                time.monotonic() - started_at,
            )
        raise
    if log_http:
        logger.info(
            "HTTP response | provider=%s | method=%s | endpoint=%s | status=%s | duration_seconds=%.3f",
            provider,
            method,
            endpoint,
            response.status_code,
            time.monotonic() - started_at,
        )
    return response


def _request_overpass(query: str) -> tuple[requests.Response, str]:
    """Use public mirrors only when an instance is temporarily unavailable."""
    last_response: requests.Response | None = None
    for endpoint in _overpass_endpoints():
        response = _send_request(
            provider="overpass",
            method="POST",
            endpoint=endpoint,
            send=lambda: requests.post(
                endpoint,
                data={"data": query},
                timeout=DEFAULT_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "User-Agent": _overpass_user_agent(),
                },
            ),
        )
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
            return response, endpoint
        last_response = response
    if last_response is None:
        raise RuntimeError("No Overpass endpoints are configured")
    last_response.raise_for_status()
    raise RuntimeError("All Overpass endpoints failed")


@code(category="retrieval")
def overpass_to_geojson(
    overpass_ql: str, output_path: str, description: str
) -> gpd.GeoDataFrame:
    """Retrieve OSM features with Overpass and persist an EPSG:4326 GeoJSON file.

    Args:
        overpass_ql: Complete spatially bounded Overpass QL query.
        output_path: Relative GeoJSON output path in the run results directory.
        description: Short human-readable description of the requested dataset.

    Returns:
        Retrieved features as an EPSG:4326 GeoDataFrame.
    """
    if not overpass_ql.strip():
        raise ValueError("overpass_ql must not be empty")
    if not description.strip():
        raise ValueError("description must not be empty")
    response, endpoint = _request_overpass(overpass_ql)
    collection = _overpass_feature_collection(response.json())
    _write_feature_collection(output_path, collection)
    get_logger().info(
        "GeoJSON retrieved | provider=overpass | endpoint=%s | description=%s | features=%d",
        endpoint,
        description,
        len(collection["features"]),
    )
    return _as_wgs84_frame(collection)


@code(category="retrieval")
def nominatim_to_geojson(
    query: str,
    output_path: str,
    description: str,
    limit: int = 10,
    country_codes: str | None = None,
) -> gpd.GeoDataFrame:
    """Search Nominatim and persist an EPSG:4326 GeoJSON FeatureCollection.

    Args:
        query: Free-text place query.
        output_path: Relative GeoJSON output path in the run results directory.
        description: Short human-readable description of the requested dataset.
        limit: Maximum number of returned features, between 1 and 50.
        country_codes: Optional comma-separated ISO 3166-1 alpha-2 country codes.

    Returns:
        Retrieved features as an EPSG:4326 GeoDataFrame.
    """
    global _last_nominatim_request
    if not query.strip():
        raise ValueError("query must not be empty")
    if not description.strip():
        raise ValueError("description must not be empty")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    elapsed = time.monotonic() - _last_nominatim_request
    if elapsed < NOMINATIM_MIN_INTERVAL_SECONDS:
        time.sleep(NOMINATIM_MIN_INTERVAL_SECONDS - elapsed)
    parameters: dict[str, str | int] = {
        "q": query,
        "format": "geojson",
        "polygon_geojson": 1,
        "limit": limit,
    }
    if country_codes:
        parameters["countrycodes"] = country_codes
    endpoint = _nominatim_endpoint()
    response = _send_request(
        provider="nominatim",
        method="GET",
        endpoint=endpoint,
        send=lambda: requests.get(
            endpoint,
            params=parameters,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            headers={
                "Accept": "application/geo+json",
                "User-Agent": _nominatim_user_agent(),
            },
        ),
    )
    _last_nominatim_request = time.monotonic()
    response.raise_for_status()
    collection = response.json()
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("Nominatim did not return a GeoJSON FeatureCollection")
    if not isinstance(collection.get("features"), list):
        raise ValueError("Nominatim GeoJSON response did not contain a features list")
    _write_feature_collection(output_path, collection)
    get_logger().info(
        "GeoJSON retrieved | provider=nominatim | endpoint=%s | description=%s | features=%d",
        endpoint,
        description,
        len(collection["features"]),
    )
    return _as_wgs84_frame(collection)


PUBLIC_RETRIEVAL_OPERATIONS = (overpass_to_geojson, nominatim_to_geojson)
