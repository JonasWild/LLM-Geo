"""Trusted geospatial/public-data operations, registered with `llm_geo.operations.registry.code`.

This is the ground-truth catalog the planner may assign directly instead of asking the coder
agent to generate code. Geo-valued inputs/outputs are real `geopandas.GeoDataFrame` objects;
retrieval operations stash provenance metadata in `gdf.attrs["provenance"]`.
"""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

from llm_geo.operations.registry import code

_USER_AGENT = "llm-geo-claude/0.1 (agentic geo-analysis demo)"

# Public OSM endpoints (Nominatim/Overpass) occasionally return transient 5xx under load.
_retry_on_server_error = retry(
    retry=lambda state: isinstance(state.outcome.exception(), requests.HTTPError)
    and state.outcome.exception().response is not None
    and state.outcome.exception().response.status_code >= 500,
    wait=wait_random_exponential(min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)


def _empty_or(features: list[dict]) -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


@code(kind="retrieval")
def read_geojson(path: str) -> gpd.GeoDataFrame:
    """Read a local GeoJSON/Shapefile path into a GeoDataFrame.

    Args:
        path: filesystem path to a GeoJSON or Shapefile.

    Returns:
        GeoDataFrame with provenance metadata in `.attrs["provenance"]`.
    """
    gdf = gpd.read_file(path)
    gdf.attrs["provenance"] = {
        "source": path, "read_at": dt.datetime.now(dt.timezone.utc).isoformat(), "crs": str(gdf.crs),
    }
    return gdf


@code(kind="retrieval")
@_retry_on_server_error
def geocode_place(query: str, limit: int = 5) -> gpd.GeoDataFrame:
    """Geocode a place name via OpenStreetMap Nominatim.

    Args:
        query: free-text place name to geocode.
        limit: maximum number of candidate matches to return.

    Returns:
        GeoDataFrame with provenance metadata in `.attrs["provenance"]`.
    """
    url = "https://nominatim.openstreetmap.org/search"
    resp = requests.get(
        url, params={"q": query, "format": "geojson", "limit": limit},
        headers={"User-Agent": _USER_AGENT}, timeout=15,
    )
    resp.raise_for_status()
    gdf = _empty_or(resp.json()["features"])
    gdf.attrs["provenance"] = {
        "source": url, "query": query, "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return gdf


@code(kind="retrieval")
@_retry_on_server_error
def overpass_query(amenity: str, bbox: str, limit: int = 20) -> gpd.GeoDataFrame:
    """Query OpenStreetMap Overpass API for point features tagged with an amenity.

    Args:
        amenity: OSM amenity tag value to match, e.g. 'cafe'.
        bbox: bounding box as 'south,west,north,east'.
        limit: maximum number of elements Overpass should return.

    Returns:
        GeoDataFrame with provenance metadata in `.attrs["provenance"]`.
    """
    url = "https://overpass-api.de/api/interpreter"
    query = f'[out:json][timeout:25];node["amenity"="{amenity}"]({bbox});out body {limit};'
    resp = requests.post(url, data={"data": query}, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"}, timeout=30)
    resp.raise_for_status()
    features = [
        {
            "type": "Feature",
            "properties": el.get("tags", {}),
            "geometry": {"type": "Point", "coordinates": [el["lon"], el["lat"]]},
        }
        for el in resp.json().get("elements", []) if "lat" in el and "lon" in el
    ]
    gdf = _empty_or(features)
    gdf.attrs["provenance"] = {
        "source": url, "query": query, "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return gdf


@code(kind="transformation")
def buffer(features: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    """Buffer every feature's geometry by a distance in CRS units.

    Args:
        features: input GeoDataFrame.
        distance: buffer distance in the GeoDataFrame's CRS units.

    Returns:
        GeoDataFrame with buffered geometries.
    """
    out = features.copy()
    out["geometry"] = out.geometry.buffer(distance)
    return out


@code(kind="transformation")
def reproject(features: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to a target CRS.

    Args:
        features: input GeoDataFrame.
        crs: target CRS, e.g. 'EPSG:3857'.

    Returns:
        GeoDataFrame reprojected to the target CRS.
    """
    return features.to_crs(crs)


@code(kind="transformation")
def filter_intersects(features: gpd.GeoDataFrame, mask_features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only features that intersect a mask GeoDataFrame.

    Args:
        features: input GeoDataFrame to filter.
        mask_features: GeoDataFrame whose union defines the keep region.

    Returns:
        GeoDataFrame containing only the intersecting features.
    """
    return features[features.intersects(mask_features.union_all())]


@code(kind="synthesis")
def summarize(features: gpd.GeoDataFrame) -> dict:
    """Summarize a GeoDataFrame into a count/bounds/geometry-type report.

    Args:
        features: input GeoDataFrame to summarize.

    Returns:
        dict with keys count, bounds, and geometry_types.
    """
    return {
        "count": int(len(features)),
        "bounds": list(features.total_bounds) if len(features) else [],
        "geometry_types": sorted(features.geom_type.unique().tolist()),
    }
