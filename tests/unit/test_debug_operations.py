"""Tests for testing-only debug operation discovery and behavior."""

from __future__ import annotations

import json
import unittest

import geopandas as gpd
from shapely.geometry import Point

from llm_geo.operations import load_all_operations


EXPECTED_DEBUG_OPERATIONS = {
    "debug_add_constant_text_column",
    "debug_buffer_features",
    "debug_calculate_area",
    "debug_calculate_length",
    "debug_centroid_points",
    "debug_clip_to_bounds",
    "debug_create_bounds",
    "debug_create_point",
    "debug_dissolve_features",
    "debug_drop_empty_geometries",
    "debug_explode_geometries",
    "debug_feature_bounds",
    "debug_feature_count",
    "debug_features_to_geojson",
    "debug_filter_features_by_text",
    "debug_geojson_to_features",
    "debug_intersect_features",
    "debug_rename_feature_column",
    "debug_reproject_features",
    "debug_sample_features",
    "debug_simplify_features",
    "debug_sort_features_by_text",
    "debug_union_features",
    "debug_validate_features",
}


class DebugOperationTests(unittest.TestCase):
    def test_debug_operations_are_discovered_and_exposed(self) -> None:
        operations = load_all_operations()
        by_name = {operation.name: operation for operation in operations}

        self.assertTrue(EXPECTED_DEBUG_OPERATIONS.issubset(by_name))
        self.assertGreaterEqual(len(EXPECTED_DEBUG_OPERATIONS), 20)
        for name in EXPECTED_DEBUG_OPERATIONS:
            self.assertEqual(by_name[name].module, "llm_geo.operations.debug.geo")

        from llm_geo.ops import debug_feature_count

        self.assertEqual(debug_feature_count(gpd.GeoDataFrame(geometry=[Point(0, 0)])), 1)

    def test_debug_operations_transform_features_and_geojson(self) -> None:
        from llm_geo.ops import (
            debug_add_constant_text_column,
            debug_feature_bounds,
            debug_features_to_geojson,
            debug_geojson_to_features,
            debug_sample_features,
        )

        features = gpd.GeoDataFrame(
            {"name": ["one", "two"]},
            geometry=[Point(0, 0), Point(2, 3)],
            crs="EPSG:4326",
        )
        enriched = debug_add_constant_text_column(features, "source", "fixture")

        self.assertEqual(debug_feature_bounds(enriched), {"minimum_x": 0.0, "minimum_y": 0.0, "maximum_x": 2.0, "maximum_y": 3.0})
        self.assertEqual(list(debug_sample_features(enriched, 1)["name"]), ["one"])
        restored = debug_geojson_to_features(debug_features_to_geojson(enriched))
        self.assertEqual(json.loads(restored.to_json())["type"], "FeatureCollection")
        self.assertEqual(len(restored), 2)