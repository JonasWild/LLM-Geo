"""Tests for the provider GeoJSON trust boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_geo.tools.data_retrieval import validate_provider_results
from llm_geo.tools.data_inspection import from_toon, to_toon
from llm_geo.tools.public_data_providers import nominatim_to_geojson, overpass_to_geojson


class ProviderResultValidationTests(unittest.TestCase):
    def test_accepts_run_scoped_feature_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "data"
            data_directory.mkdir()
            source_path = data_directory / "places.geojson"
            source_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": []}),
                encoding="utf-8",
            )
            result = to_toon(
                {
                    "description": "Retrieved places",
                    "location": str(source_path),
                    "provider": "fake_provider",
                    "request": {"query": "places"},
                }
            )

            sources = validate_provider_results([result], data_directory)

            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].format, "GeoJSON")
            self.assertEqual(sources[0].location, str(source_path.resolve()))

    def test_rejects_path_outside_run_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_directory = root / "data"
            data_directory.mkdir()
            source_path = root / "outside.geojson"
            source_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": []}),
                encoding="utf-8",
            )
            result = to_toon(
                {
                    "description": "Outside file",
                    "location": str(source_path),
                    "provider": "fake_provider",
                }
            )

            with self.assertRaisesRegex(ValueError, "outside the run data directory"):
                validate_provider_results([result], data_directory)

    def test_rejects_non_feature_collection_geojson(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "data"
            data_directory.mkdir()
            source_path = data_directory / "feature.geojson"
            source_path.write_text(
                json.dumps({"type": "Feature", "geometry": None, "properties": {}}),
                encoding="utf-8",
            )
            result = to_toon(
                {
                    "description": "Invalid shape",
                    "location": str(source_path),
                    "provider": "fake_provider",
                }
            )

            with self.assertRaisesRegex(ValueError, "FeatureCollection"):
                validate_provider_results([result], data_directory)


class PublicProviderToolTests(unittest.TestCase):
    @patch("llm_geo.tools.public_data_providers.requests.post")
    def test_overpass_converts_nodes_and_ways_to_feature_collection(
        self, post: Mock
    ) -> None:
        response = Mock()
        response.json.return_value = {
            "elements": [
                {"type": "node", "id": 1, "lat": 52.52, "lon": 13.405},
                {
                    "type": "way",
                    "id": 2,
                    "tags": {"landuse": "residential"},
                    "geometry": [
                        {"lat": 52.5, "lon": 13.4},
                        {"lat": 52.5, "lon": 13.5},
                        {"lat": 52.6, "lon": 13.5},
                        {"lat": 52.5, "lon": 13.4},
                    ],
                },
            ]
        }
        post.return_value = response
        configured_endpoint = "https://overpass.internal.example/api/interpreter"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "overpass.geojson"

            with (
                patch.dict("os.environ", {"OVERPASS_URL": configured_endpoint}),
                patch(
                    "llm_geo.tools.public_data_providers.http_logging_enabled",
                    return_value=True,
                ),
                patch("llm_geo.tools.public_data_providers.get_logger") as get_logger,
            ):
                result = from_toon(
                    overpass_to_geojson.invoke(
                        {
                            "overpass_ql": "node(1);out geom;",
                            "output_path": str(output_path),
                            "description": "Sample OSM data",
                        }
                    )
                )

            collection = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["provider"], "overpass")
            self.assertEqual(result["request"]["endpoint"], configured_endpoint)
            self.assertEqual(collection["type"], "FeatureCollection")
            self.assertEqual(collection["features"][0]["geometry"]["type"], "Point")
            self.assertEqual(collection["features"][1]["geometry"]["type"], "Polygon")
            post.assert_called_once()
            self.assertEqual(post.call_args.args[0], configured_endpoint)
            self.assertIn("User-Agent", post.call_args.kwargs["headers"])
            self.assertEqual(get_logger.return_value.info.call_count, 2)

    @patch("llm_geo.tools.public_data_providers.requests.get")
    def test_nominatim_persists_returned_feature_collection(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"display_name": "Magdeburg"},
                    "geometry": {"type": "Point", "coordinates": [11.62, 52.12]},
                }
            ],
        }
        get.return_value = response
        configured_endpoint = "https://nominatim.internal.example/search"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nominatim.geojson"
            with patch.dict(
                "os.environ",
                {
                    "NOMINATIM_USER_AGENT": "LLM-GEO tests (contact: tests@example.com)",
                    "NOMINATIM_URL": configured_endpoint,
                },
            ), patch(
                "llm_geo.tools.public_data_providers.http_logging_enabled",
                return_value=True,
            ), patch("llm_geo.tools.public_data_providers.get_logger") as get_logger:
                result = from_toon(
                    nominatim_to_geojson.invoke(
                        {
                            "query": "Magdeburg, Germany",
                            "output_path": str(output_path),
                            "description": "Magdeburg search result",
                            "country_codes": "de",
                        }
                    )
                )

            collection = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["provider"], "nominatim")
            self.assertEqual(result["request"]["endpoint"], configured_endpoint)
            self.assertEqual(result["request"]["country_codes"], "de")
            self.assertEqual(collection["type"], "FeatureCollection")
            self.assertEqual(len(collection["features"]), 1)
            self.assertEqual(get.call_args.args[0], configured_endpoint)
            self.assertEqual(get.call_args.kwargs["params"]["format"], "geojson")
            self.assertEqual(get_logger.return_value.info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
