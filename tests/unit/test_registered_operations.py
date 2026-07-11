"""Tests for strictly typed trusted Python operation registration."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

from llm_geo.operations import code, registered_operations
from llm_geo.tools.workflow_graph import (
    operation_contract,
    registered_operation_bridge,
    validate_workflow_plan,
)
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


@code(category="retrieval")
def retrieve_fixture(query: str, limit: int = 10) -> str:
    """Retrieve a fixture dataset.

    Args:
        query: Dataset query.
        limit: Maximum result count.

    Returns:
        Retrieved fixture identifier.
    """
    return f"{query}:{limit}"


class RegisteredOperationTests(unittest.TestCase):
    def test_public_provider_operations_import_in_a_clean_process(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from llm_geo.ops import nominatim_to_geojson, overpass_to_geojson",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_decorator_derives_catalog_from_types_and_docstring(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "integer_to_text"
        )

        self.assertEqual(operation.id, "integer_to_text")
        self.assertEqual(operation.qualified_id, f"{__name__}.integer_to_text")
        self.assertEqual(
            operation.catalog_entry()["import"],
            "from llm_geo.ops import integer_to_text",
        )
        from llm_geo.ops import integer_to_text as public_integer_to_text

        self.assertIs(public_integer_to_text, integer_to_text)
        self.assertEqual(operation.inputs, (("value", "int", "Integer to convert."),))
        self.assertEqual(operation.output_type, "str")
        self.assertEqual(operation.output_description, "Text representation of the integer.")
        self.assertEqual(operation.category, "transformation")

    def test_decorator_exposes_retrieval_category_in_catalog(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "retrieve_fixture"
        )

        self.assertEqual(operation.category, "retrieval")
        self.assertEqual(operation.catalog_entry()["category"], "retrieval")

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
                PlanNode(id="source", kind="data", description="Input", data_path="input.geojson", implementation="generated"),
                PlanNode(
                    id="convert",
                    kind="operation",
                    description="Convert input",
                    implementation="registered",
                    registered_operation_id=operation.id,
                ),
                PlanNode(id="result", kind="data", description="Output", implementation="generated"),
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

    def test_zero_input_registered_operation_accepts_literal_arguments(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "retrieve_fixture"
        )
        plan = WorkflowPlan(
            rationale="Retrieve through a trusted operation.",
            nodes=[
                PlanNode(
                    id="retrieve",
                    kind="operation",
                    description="Retrieve data",
                    implementation="registered",
                    registered_operation_id=operation.id,
                    literal_arguments={"query": "parks"},
                ),
                PlanNode(id="features", kind="data", description="Retrieved data", implementation="generated"),
            ],
            edges=[PlanEdge(source="retrieve", target="features")],
        )

        self.assertEqual(validate_workflow_plan(plan, [], [operation]), [])
        legacy_plan = plan.model_copy(deep=True)
        legacy_plan.nodes[0].registered_operation_id = operation.qualified_id
        self.assertEqual(validate_workflow_plan(legacy_plan, [], [operation]), [])
        self.assertEqual(
            registered_operation_bridge(plan, "retrieve", operation),
            "from llm_geo.ops import retrieve_fixture\n\n"
            "def retrieve():\n"
            "    return retrieve_fixture(query='parks')",
        )

    def test_registered_operation_rejects_unknown_and_missing_literals(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "retrieve_fixture"
        )
        plan = WorkflowPlan(
            rationale="Invalid literal bindings.",
            nodes=[
                PlanNode(
                    id="retrieve",
                    kind="operation",
                    description="Retrieve data",
                    implementation="registered",
                    registered_operation_id=operation.id,
                    literal_arguments={"unexpected": True},
                ),
                PlanNode(id="features", kind="data", description="Retrieved data", implementation="generated"),
            ],
            edges=[PlanEdge(source="retrieve", target="features")],
        )

        issues = validate_workflow_plan(plan, [], [operation])

        self.assertTrue(any("unknown literal arguments" in issue for issue in issues))
        self.assertTrue(any("missing required arguments: query" in issue for issue in issues))

    def test_retrieval_operation_rejects_graph_inputs(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "retrieve_fixture"
        )
        plan = WorkflowPlan(
            rationale="Invalid retrieval dependency.",
            nodes=[
                PlanNode(
                    id="query_data",
                    kind="data",
                    description="Incorrect query data node",
                    data_path="query.txt",
                    implementation="generated",
                ),
                PlanNode(
                    id="retrieve",
                    kind="operation",
                    description="Retrieve data",
                    implementation="registered",
                    registered_operation_id=operation.id,
                ),
                PlanNode(
                    id="features",
                    kind="data",
                    description="Retrieved data",
                    implementation="generated",
                ),
            ],
            edges=[
                PlanEdge(source="query_data", target="retrieve"),
                PlanEdge(source="retrieve", target="features"),
            ],
        )

        issues = validate_workflow_plan(plan, [], [operation])

        self.assertTrue(any("must be a root operation" in issue for issue in issues))

    def test_retrieval_output_path_must_match_its_data_node(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "retrieve_fixture"
        )
        plan = WorkflowPlan(
            rationale="Retrieval cannot directly produce another file format.",
            nodes=[
                PlanNode(
                    id="retrieve",
                    kind="operation",
                    description="Retrieve data",
                    implementation="registered",
                    registered_operation_id=operation.id,
                    literal_arguments={"query": "parks", "output_path": "parks.geojson"},
                ),
                PlanNode(
                    id="image",
                    kind="data",
                    description="Incorrect image output",
                    data_path="parks.png",
                    implementation="generated",
                ),
            ],
            edges=[PlanEdge(source="retrieve", target="image")],
        )

        # Use a catalog-shaped retrieval operation that exposes output_path.
        operation_with_path = operation.__class__(
            **{
                **operation.__dict__,
                "inputs": (
                    ("query", "str", "Dataset query."),
                    ("output_path", "str", "Output path."),
                ),
            }
        )
        issues = validate_workflow_plan(plan, [], [operation_with_path])

        self.assertTrue(any("required conversion or rendering" in issue for issue in issues))

    def test_root_data_node_requires_existing_source_path(self) -> None:
        operation = next(
            item for item in registered_operations() if item.name == "integer_to_text"
        )
        plan = WorkflowPlan(
            rationale="Invalid origin-less data.",
            nodes=[
                PlanNode(
                    id="source",
                    kind="data",
                    description="Origin-less input",
                    implementation="generated",
                ),
                PlanNode(
                    id="convert",
                    kind="operation",
                    description="Convert input",
                    implementation="registered",
                    registered_operation_id=operation.id,
                ),
                PlanNode(
                    id="result",
                    kind="data",
                    description="Output",
                    implementation="generated",
                ),
            ],
            edges=[
                PlanEdge(source="source", target="convert"),
                PlanEdge(source="convert", target="result"),
            ],
        )

        issues = validate_workflow_plan(plan, [], [operation])

        self.assertTrue(any("no producing operation and no existing source path" in issue for issue in issues))

    def test_operation_requires_explicit_implementation(self) -> None:
        with self.assertRaisesRegex(ValidationError, "implementation"):
            PlanNode(id="transform", kind="operation", description="Transform data")

    def test_generated_operation_requires_reason(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "why no registered operation applies"
        ):
            PlanNode(
                id="transform",
                kind="operation",
                description="Transform data",
                implementation="generated",
            )

        node = PlanNode(
            id="transform",
            kind="operation",
            description="Transform data",
            implementation="generated",
            generation_reason="No registered operation performs this transformation.",
        )
        self.assertEqual(node.implementation, "generated")

    def test_generated_operation_exposes_literals_as_function_parameters(self) -> None:
        plan = WorkflowPlan(
            rationale="Buffer input features by a task-defined distance.",
            nodes=[
                PlanNode(
                    id="features",
                    kind="data",
                    description="Existing features",
                    data_path="features.geojson",
                    implementation="generated",
                ),
                PlanNode(
                    id="buffer_features",
                    kind="operation",
                    description="Buffer features",
                    implementation="generated",
                    literal_arguments={"distance_meters": 25},
                    generation_reason="No registered buffer operation is available.",
                ),
                PlanNode(
                    id="buffered",
                    kind="data",
                    description="Buffered features",
                    implementation="generated",
                ),
            ],
            edges=[
                PlanEdge(source="features", target="buffer_features"),
                PlanEdge(source="buffer_features", target="buffered"),
            ],
        )

        self.assertEqual(validate_workflow_plan(plan, [], []), [])
        contract = operation_contract(plan, "buffer_features")
        self.assertEqual(contract["inputs"], ["features"])
        self.assertEqual(contract["literal_arguments"], {"distance_meters": 25})
        self.assertEqual(
            contract["signature"],
            "def buffer_features(features, distance_meters):",
        )

    def test_generated_operation_rejects_invalid_literal_parameter_name(self) -> None:
        plan = WorkflowPlan(
            rationale="Invalid generated parameter.",
            nodes=[
                PlanNode(
                    id="features",
                    kind="data",
                    description="Existing features",
                    data_path="features.geojson",
                    implementation="generated",
                ),
                PlanNode(
                    id="transform",
                    kind="operation",
                    description="Transform features",
                    implementation="generated",
                    literal_arguments={"distance-meters": 25},
                    generation_reason="No registered operation applies.",
                ),
                PlanNode(
                    id="result",
                    kind="data",
                    description="Result",
                    implementation="generated",
                ),
            ],
            edges=[
                PlanEdge(source="features", target="transform"),
                PlanEdge(source="transform", target="result"),
            ],
        )

        issues = validate_workflow_plan(plan, [], [])

        self.assertTrue(any("invalid literal parameter name" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
