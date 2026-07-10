"""Tests for system and observed-execution Mermaid artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_geo.tools.mermaid_diagnostics import (
    execution_event,
    execution_mermaid,
    write_execution_graph_artifacts,
    write_system_graph_artifacts,
)


class MermaidDiagnosticsTests(unittest.TestCase):
    def test_execution_events_count_repeated_nodes(self) -> None:
        first = execution_event([], "plan_workflow", "plan_created")
        second = execution_event([first], "validate_plan", "plan_invalid")
        third = execution_event(
            [first, second], "plan_workflow", "plan_created"
        )

        self.assertEqual(first["occurrence"], 1)
        self.assertEqual(second["outcome"], "failure")
        self.assertEqual(third["occurrence"], 2)
        self.assertEqual(third["sequence"], 3)

    def test_execution_mermaid_preserves_route_and_escapes_labels(self) -> None:
        trace = [
            execution_event([], 'retrieve_"sources"', "sources_retrieved"),
        ]

        source = execution_mermaid(trace)

        self.assertIn("start --> event_1", source)
        self.assertIn("event_1 --> finish", source)
        self.assertIn("retrieve_&quot;sources&quot; #1", source)
        self.assertIn("sources_retrieved", source)

    @patch("llm_geo.tools.mermaid_diagnostics.render_mermaid_png")
    def test_execution_writer_keeps_source_when_rendering_is_unavailable(
        self, render: Mock
    ) -> None:
        render.return_value = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            artifacts = write_execution_graph_artifacts([], root)

            source_path = root / "workflow" / "execution.mmd"
            self.assertEqual(artifacts, [str(source_path)])
            self.assertTrue(source_path.is_file())
            self.assertFalse((root / "workflow" / "execution.png").exists())

    @patch("llm_geo.tools.mermaid_diagnostics.render_mermaid_png")
    def test_system_writer_uses_compiled_graph_mermaid(self, render: Mock) -> None:
        render.return_value = True
        drawable = Mock()
        drawable.draw_mermaid.return_value = "flowchart TD\n    a --> b\n"
        graph = Mock()
        graph.get_graph.return_value = drawable
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            artifacts = write_system_graph_artifacts(graph, root)

            source_path = root / "workflow" / "system.mmd"
            png_path = root / "workflow" / "system.png"
            self.assertEqual(artifacts, [str(source_path), str(png_path)])
            self.assertEqual(
                source_path.read_text(encoding="utf-8"),
                "flowchart TD\n    a --> b\n",
            )
            render.assert_called_once_with(source_path, png_path)


if __name__ == "__main__":
    unittest.main()
