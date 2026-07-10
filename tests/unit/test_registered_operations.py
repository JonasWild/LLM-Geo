"""Tests for strictly typed trusted Python operation registration."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from llm_geo.operations import code, registered_operations
from llm_geo.tools.workflow_graph import validate_workflow_plan
from llm_geo.utils.models import DataSource, PlanEdge, PlanNode, WorkflowPlan


@code
def integer_to_text(value: int) -> str:
    """Convert an integer into text.

    Args:
        value: Integer to convert.

    Returns:
        Text representation of the integer.
    """
    return str(value)


class RegisteredOperationTests(unittest.TestCase):
    def test_decorator_derives_catalog_from_types_and_docstring(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "integer_to_text"
        )

        self.assertEqual(operation.id, f"{__name__}.integer_to_text")
        self.assertEqual(operation.inputs, (("value", "int", "Integer to convert."),))
        self.assertEqual(operation.output_type, "str")
        self.assertEqual(operation.output_description, "Text representation of the integer.")

    def test_decorator_rejects_untyped_functions(self) -> None:
        def incomplete(value):
            """Return a value.

            Args:
                value: Value to return.

            Returns:
                Returned value.
            """
            return value

        incomplete.__qualname__ = "incomplete"
        with self.assertRaisesRegex(TypeError, "concrete return type"):
            code(incomplete)

    def test_registered_plan_requires_matching_port_count(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "integer_to_text"
        )
        plan = WorkflowPlan(
            rationale="Use the trusted conversion function.",
            nodes=[
                PlanNode(id="source", kind="data", description="Input", data_path="input.geojson"),
                PlanNode(
                    id="convert",
                    kind="operation",
                    description="Convert input",
                    implementation="registered",
                    registered_operation_id=operation.id,
                ),
                PlanNode(id="result", kind="data", description="Output"),
            ],
            edges=[PlanEdge(source="source", target="convert"), PlanEdge(source="convert", target="result")],
        )
        sources = [
            DataSource(
                description="Input source",
                location="input.geojson",
                provider="test",
            )
        ]

        self.assertEqual(validate_workflow_plan(plan, sources, [operation]), [])

    def test_data_node_rejects_operation_implementation_metadata(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "Data nodes cannot select an implementation"
        ):
            PlanNode(
                id="source",
                kind="data",
                description="Input",
                implementation="registered",
                registered_operation_id="example.operation",
            )


if __name__ == "__main__":
    unittest.main()
