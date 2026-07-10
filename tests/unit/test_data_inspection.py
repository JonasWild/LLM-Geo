"""Tests for GeoJSON source inspection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_geo.tools.data_inspection import from_toon, inspect_source, to_toon
from llm_geo.utils.models import DataSource


class DataInspectionTests(unittest.TestCase):
    def test_toon_round_trip_preserves_prompt_data(self) -> None:
        data = {
            "sources": [
                {
                    "description": "Magdeburg",
                    "metadata": {"feature_count": 1, "columns": ["name"]},
                }
            ]
        }

        self.assertEqual(from_toon(to_toon(data)), data)

    def test_inspect_source_attaches_geojson_vector_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "places.geojson"
            source_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"name": "Magdeburg"},
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [11.6276, 52.1205],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            inspected = inspect_source(
                DataSource(
                    description="Retrieved places",
                    location=str(source_path),
                    provider="test_provider",
                )
            )

            self.assertIsNone(inspected.inspection_error)
            self.assertEqual(inspected.metadata["feature_count"], 1)
            self.assertEqual(inspected.metadata["geometry_types"], ["Point"])
            self.assertEqual(inspected.metadata["columns"], ["name"])
            self.assertEqual(inspected.metadata["sample"], [{"name": "Magdeburg"}])

    def test_inspect_source_retains_failure_for_planner(self) -> None:
        inspected = inspect_source(
            DataSource(
                description="Missing source",
                location="missing.geojson",
                provider="test_provider",
            )
        )

        self.assertEqual(inspected.metadata, {})
        self.assertIsNotNone(inspected.inspection_error)
        self.assertIn("DataSourceError", inspected.inspection_error)


if __name__ == "__main__":
    unittest.main()
