"""Offline tests for direct OpenAPI operation generation and invocation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_geo.operations.openapi import generator
from llm_geo.operations.openapi.generator import generate_openapi_operations
from llm_geo.operations.openapi.parser import parse_openapi
from llm_geo.operations.openapi.runtime import invoke_json
from llm_geo.operations.openapi.renderer import render_module


FIXTURE = Path(__file__).parents[1] / "fixtures" / "openapi" / "demo.json"


def load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class OpenAPIParserTests(unittest.TestCase):
    def test_parser_normalizes_names_types_defaults_and_skips_binary(self) -> None:
        result = parse_openapi(load_fixture())

        self.assertEqual(
            [operation.function_name for operation in result.operations],
            ["get_area", "buffer_geometry"],
        )
        get_area = result.operations[0]
        self.assertEqual(get_area.return_annotation, "dict[str, Any]")
        self.assertEqual(
            [(item.python_name, item.annotation, item.required) for item in get_area.parameters],
            [
                ("area_id", "int", True),
                ("include_geometry", "bool | None", False),
            ],
        )
        buffer_geometry = result.operations[1]
        self.assertEqual(buffer_geometry.parameters[-1].python_name, "segments")
        self.assertEqual(buffer_geometry.parameters[-1].default, 8)
        self.assertEqual(result.skipped[0]["operation_id"], "download_file")

    def test_renderer_emits_registry_compatible_docstrings_and_decorators(self) -> None:
        parsed = parse_openapi(load_fixture())
        source = render_module(
            parsed.operations,
            service="demo",
            default_base_url="https://geo.example.test",
        )

        self.assertEqual(source.count("@code"), 2)
        self.assertIn("area_id: Identifier of the mapped area.", source)
        self.assertNotIn("area_id (int):", source)
        compile(source, "generated_demo.py", "exec")


class OpenAPIGeneratorTests(unittest.TestCase):
    def test_generation_validates_registers_and_skips_unchanged_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated"
            first = generate_openapi_operations(
                load_fixture(),
                service="demo-service",
                default_base_url="https://geo.example.test",
                output_directory=output,
                source=str(FIXTURE),
            )
            second = generate_openapi_operations(
                load_fixture(),
                service="demo-service",
                default_base_url="https://geo.example.test",
                output_directory=output,
                source=str(FIXTURE),
            )

            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            self.assertEqual(first.operation_ids, ("get_area", "buffer_geometry"))
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["skipped"][0]["operation_id"], "download_file")
            self.assertIn("@code", first.module_path.read_text(encoding="utf-8"))

    def test_failed_validation_preserves_existing_generated_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated"
            output.mkdir(parents=True)
            existing = output / "demo.py"
            existing.write_text("# previous valid module\n", encoding="utf-8")

            with patch.object(
                generator, "_validate_generated_module", side_effect=RuntimeError("invalid")
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid"):
                    generate_openapi_operations(
                        load_fixture(),
                        service="demo",
                        default_base_url="https://geo.example.test",
                        output_directory=output,
                        force=True,
                    )

            self.assertEqual(existing.read_text(encoding="utf-8"), "# previous valid module\n")


class OpenAPIRuntimeTests(unittest.TestCase):
    def test_invoke_json_renders_request_and_uses_service_credentials(self) -> None:
        response = Mock()
        response.json.return_value = {"type": "Feature"}
        response.raise_for_status.return_value = None
        session = Mock()
        session.request.return_value = response
        environment = {
            "LLM_GEO_OPENAPI_DEMO_URL": "https://override.example.test/",
            "LLM_GEO_OPENAPI_DEMO_API_KEY": "secret",
        }

        with patch.dict(os.environ, environment, clear=False), patch(
            "llm_geo.operations.openapi.runtime._session", return_value=session
        ):
            result = invoke_json(
                service="demo",
                method="GET",
                path="/areas/{area-id}",
                default_base_url="https://default.example.test",
                path_parameters={"area-id": "one/two"},
                query_parameters={"include": True, "omit": None},
            )

        self.assertEqual(result, {"type": "Feature"})
        session.request.assert_called_once_with(
            method="GET",
            url="https://override.example.test/areas/one%2Ftwo",
            params={"include": True},
            headers={"Authorization": "Bearer secret"},
            json=None,
            timeout=30.0,
        )


if __name__ == "__main__":
    unittest.main()
