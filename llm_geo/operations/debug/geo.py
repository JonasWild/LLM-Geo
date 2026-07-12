"""Deterministic geospatial operations used only for registry testing."""

from __future__ import annotations

import json

import geopandas as gpd
from shapely.geometry import Point, box

from llm_geo.operations import code


@code
def debug_create_point(longitude: float, latitude: float, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Create a one-feature GeoDataFrame containing a point.

    Args:
        longitude: X coordinate of the point.
        latitude: Y coordinate of the point.
        crs: Coordinate reference system assigned to the result.

    Returns:
        GeoDataFrame with one Point geometry.
    """
    return gpd.GeoDataFrame(geometry=[Point(longitude, latitude)], crs=crs)


@code
def debug_create_bounds(minimum_x: float, minimum_y: float, maximum_x: float, maximum_y: float, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Create a one-feature GeoDataFrame containing a rectangular boundary.

    Args:
        minimum_x: Minimum X coordinate of the rectangle.
        minimum_y: Minimum Y coordinate of the rectangle.
        maximum_x: Maximum X coordinate of the rectangle.
        maximum_y: Maximum Y coordinate of the rectangle.
        crs: Coordinate reference system assigned to the result.

    Returns:
        GeoDataFrame with one Polygon geometry.
    """
    return gpd.GeoDataFrame(geometry=[box(minimum_x, minimum_y, maximum_x, maximum_y)], crs=crs)


@code
def debug_feature_bounds(features: gpd.GeoDataFrame) -> dict[str, float]:
    """Return the total bounds of all feature geometries.

    Args:
        features: Geospatial features whose extent is measured.

    Returns:
        Dictionary containing minimum and maximum X and Y coordinates.
    """
    minimum_x, minimum_y, maximum_x, maximum_y = features.total_bounds
    return {"minimum_x": minimum_x, "minimum_y": minimum_y, "maximum_x": maximum_x, "maximum_y": maximum_y}


@code
def debug_reproject_features(features: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject geospatial features into a target coordinate reference system.

    Args:
        features: Geospatial features with an assigned coordinate reference system.
        target_crs: Coordinate reference system for the returned features.

    Returns:
        GeoDataFrame reprojected into the target coordinate reference system.
    """
    return features.to_crs(target_crs)


@code
def debug_buffer_features(features: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    """Buffer every feature geometry by a fixed coordinate-unit distance.

    Args:
        features: Geospatial features to buffer.
        distance: Buffer distance in the coordinate units of the input CRS.

    Returns:
        GeoDataFrame with buffered geometries and preserved attributes.
    """
    result = features.copy()
    result.geometry = result.geometry.buffer(distance)
    return result


@code
def debug_simplify_features(features: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """Simplify every feature geometry using a coordinate-unit tolerance.

    Args:
        features: Geospatial features to simplify.
        tolerance: Simplification tolerance in the coordinate units of the input CRS.

    Returns:
        GeoDataFrame with simplified geometries and preserved attributes.
    """
    result = features.copy()
    result.geometry = result.geometry.simplify(tolerance)
    return result


@code
def debug_centroid_points(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace every geometry with its centroid point.

    Args:
        features: Geospatial features whose centroids are calculated.

    Returns:
        GeoDataFrame containing centroid Point geometries and preserved attributes.
    """
    result = features.copy()
    result.geometry = result.geometry.centroid
    return result


@code
def debug_explode_geometries(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Expand multi-part geometries into one row per individual geometry.

    Args:
        features: Geospatial features that may contain multi-part geometries.

    Returns:
        GeoDataFrame with individual geometry parts as separate rows.
    """
    return features.explode(index_parts=False, ignore_index=True)


@code
def debug_dissolve_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve all feature geometries into one combined feature.

    Args:
        features: Geospatial features to combine.

    Returns:
        One-row GeoDataFrame containing the dissolved geometry.
    """
    return features.dissolve()


@code
def debug_clip_to_bounds(features: gpd.GeoDataFrame, minimum_x: float, minimum_y: float, maximum_x: float, maximum_y: float) -> gpd.GeoDataFrame:
    """Clip feature geometries to a rectangular coordinate extent.

    Args:
        features: Geospatial features to clip.
        minimum_x: Minimum X coordinate of the clipping rectangle.
        minimum_y: Minimum Y coordinate of the clipping rectangle.
        maximum_x: Maximum X coordinate of the clipping rectangle.
        maximum_y: Maximum Y coordinate of the clipping rectangle.

    Returns:
        GeoDataFrame containing non-empty geometry intersections with the rectangle.
    """
    result = features.copy()
    result.geometry = result.geometry.intersection(box(minimum_x, minimum_y, maximum_x, maximum_y))
    return result.loc[~result.geometry.is_empty].copy()


@code
def debug_filter_features_by_text(features: gpd.GeoDataFrame, column: str, value: str) -> gpd.GeoDataFrame:
    """Keep features whose selected attribute equals a text value.

    Args:
        features: Geospatial features to filter.
        column: Attribute column used for the comparison.
        value: Text value that matching features must contain.

    Returns:
        GeoDataFrame containing only matching features.
    """
    return features.loc[features[column].astype(str) == value].copy()


@code
def debug_drop_empty_geometries(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove features that have missing or empty geometries.

    Args:
        features: Geospatial features to clean.

    Returns:
        GeoDataFrame containing only non-empty geometries.
    """
    return features.loc[features.geometry.notna() & ~features.geometry.is_empty].copy()


@code
def debug_rename_feature_column(features: gpd.GeoDataFrame, source_column: str, target_column: str) -> gpd.GeoDataFrame:
    """Rename one non-geometry attribute column.

    Args:
        features: Geospatial features whose attributes are renamed.
        source_column: Existing attribute column name.
        target_column: Replacement attribute column name.

    Returns:
        GeoDataFrame with the requested attribute column renamed.
    """
    return features.rename(columns={source_column: target_column})


@code
def debug_add_constant_text_column(features: gpd.GeoDataFrame, column: str, value: str) -> gpd.GeoDataFrame:
    """Add or replace an attribute column with a constant text value.

    Args:
        features: Geospatial features to augment.
        column: Attribute column to add or replace.
        value: Text value assigned to every feature.

    Returns:
        GeoDataFrame with the constant attribute column.
    """
    result = features.copy()
    result[column] = value
    return result


@code
def debug_feature_count(features: gpd.GeoDataFrame) -> int:
    """Count the number of feature rows.

    Args:
        features: Geospatial features to count.

    Returns:
        Number of rows in the GeoDataFrame.
    """
    return len(features)


@code
def debug_calculate_area(features: gpd.GeoDataFrame, column: str = "debug_area") -> gpd.GeoDataFrame:
    """Calculate geometry areas and store them in an attribute column.

    Args:
        features: Geospatial features whose geometry areas are calculated.
        column: Attribute column that receives the calculated areas.

    Returns:
        GeoDataFrame with an area attribute column.
    """
    result = features.copy()
    result[column] = result.geometry.area
    return result


@code
def debug_calculate_length(features: gpd.GeoDataFrame, column: str = "debug_length") -> gpd.GeoDataFrame:
    """Calculate geometry lengths and store them in an attribute column.

    Args:
        features: Geospatial features whose geometry lengths are calculated.
        column: Attribute column that receives the calculated lengths.

    Returns:
        GeoDataFrame with a length attribute column.
    """
    result = features.copy()
    result[column] = result.geometry.length
    return result


@code
def debug_intersect_features(left_features: gpd.GeoDataFrame, right_features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return geometry intersections between two feature collections.

    Args:
        left_features: First geospatial feature collection.
        right_features: Second geospatial feature collection with the same CRS.

    Returns:
        GeoDataFrame containing intersecting geometry portions.
    """
    return gpd.overlay(left_features, right_features, how="intersection")


@code
def debug_union_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge all geometries into a single union geometry.

    Args:
        features: Geospatial features whose geometries are unioned.

    Returns:
        One-row GeoDataFrame containing the unioned geometry.
    """
    return gpd.GeoDataFrame(geometry=[features.geometry.unary_union], crs=features.crs)


@code
def debug_geojson_to_features(geojson: str) -> gpd.GeoDataFrame:
    """Parse a GeoJSON feature collection string into geospatial features.

    Args:
        geojson: GeoJSON FeatureCollection encoded as text.

    Returns:
        GeoDataFrame created from the GeoJSON features.
    """
    payload = json.loads(geojson)
    return gpd.GeoDataFrame.from_features(payload["features"])


@code
def debug_features_to_geojson(features: gpd.GeoDataFrame) -> str:
    """Serialize geospatial features as a GeoJSON feature collection string.

    Args:
        features: Geospatial features to serialize.

    Returns:
        GeoJSON FeatureCollection encoded as text.
    """
    return features.to_json()


@code
def debug_validate_features(features: gpd.GeoDataFrame) -> dict[str, int]:
    """Summarize feature count and geometry validity for test assertions.

    Args:
        features: Geospatial features to inspect.

    Returns:
        Dictionary containing total, valid, invalid, and empty feature counts.
    """
    empty_count = int(features.geometry.isna().sum() + features.geometry.fillna(Point()).is_empty.sum())
    valid_count = int(features.geometry.notna().sum() and features.geometry.dropna().is_valid.sum())
    return {"total": len(features), "valid": valid_count, "invalid": len(features) - valid_count - empty_count, "empty": empty_count}


@code
def debug_sample_features(features: gpd.GeoDataFrame, count: int) -> gpd.GeoDataFrame:
    """Return the first requested number of feature rows.

    Args:
        features: Geospatial features to sample.
        count: Maximum number of rows returned from the beginning of the data.

    Returns:
        GeoDataFrame containing at most the requested number of rows.
    """
    return features.head(max(count, 0)).copy()


@code
def debug_sort_features_by_text(features: gpd.GeoDataFrame, column: str) -> gpd.GeoDataFrame:
    """Sort geospatial features by a text representation of one attribute.

    Args:
        features: Geospatial features to sort.
        column: Attribute column used as the sort key.

    Returns:
        GeoDataFrame sorted by the selected attribute value.
    """
    return features.assign(_debug_sort_key=features[column].astype(str)).sort_values("_debug_sort_key").drop(columns="_debug_sort_key")