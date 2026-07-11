"""Tests for token-efficient program composition and repair helpers."""

from __future__ import annotations

import unittest

from llm_geo.system import (
    apply_code_replacements,
    compose_program,
    resolve_code_repair,
)
from llm_geo.tools.workflow_graph import (
    operation_context,
    render_main,
    validate_operation_code,
    validate_workflow_plan,
)
from llm_geo.utils.models import (
    CodeRepair,
    CodeReplacement,
    PlanEdge,
    PlanNode,
    WorkflowPlan,
)


class ProgramCompositionTests(unittest.TestCase):
    def test_composition_preserves_reviewed_function_text(self) -> None:
        function = "def transform(source):\n    return source.copy()\n"

        program = compose_program(
            "import json",
            [function],
            "def main():\n    return transform({})\n\nmain()",
        )

        self.assertIn(function, program)
        compile(program, "<test>", "exec")

    def test_composition_rejects_invalid_glue(self) -> None:
        with self.assertRaises(SyntaxError):
            compose_program("", ["def transform():\n    return 1"], "not python:")


class CodeRepairTests(unittest.TestCase):
    def test_applies_one_exact_replacement(self) -> None:
        patched = apply_code_replacements(
            "value = 1\nprint(value)\n",
            [CodeReplacement(old="value = 1", new="value = 2")],
        )

        self.assertEqual(patched, "value = 2\nprint(value)\n")

    def test_rejects_missing_or_ambiguous_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0"):
            apply_code_replacements(
                "value = 1\n",
                [CodeReplacement(old="value = 2", new="value = 3")],
            )
        with self.assertRaisesRegex(ValueError, "found 2"):
            apply_code_replacements(
                "value = 1\nvalue = 1\n",
                [CodeReplacement(old="value = 1", new="value = 2")],
            )

    def test_rejects_syntactically_invalid_patch(self) -> None:
        with self.assertRaises(SyntaxError):
            apply_code_replacements(
                "value = 1\n",
                [CodeReplacement(old="value = 1", new="value =")],
            )

    def test_complete_code_is_supported_as_fallback(self) -> None:
        result = resolve_code_repair(
            "value = 1\n",
            CodeRepair(complete_code="value = 3\n"),
        )

        self.assertEqual(result, "value = 3\n")


class OperationContextTests(unittest.TestCase):
    def test_context_contains_only_adjacent_operation_contracts(self) -> None:
        def data(node_id: str) -> PlanNode:
            return PlanNode(
                id=node_id,
                kind="data",
                description=node_id,
                implementation="generated",
            )

        def operation(node_id: str) -> PlanNode:
            return PlanNode(
                id=node_id,
                kind="operation",
                description=node_id,
                implementation="generated",
                generation_reason="No registered operation matches.",
            )

        plan = WorkflowPlan(
            rationale="test",
            nodes=[
                data("source"),
                operation("first"),
                data("middle"),
                operation("second"),
                data("result"),
                operation("third"),
                data("final"),
            ],
            edges=[
                PlanEdge(source="source", target="first"),
                PlanEdge(source="first", target="middle"),
                PlanEdge(source="middle", target="second"),
                PlanEdge(source="second", target="result"),
                PlanEdge(source="result", target="third"),
                PlanEdge(source="third", target="final"),
            ],
        )

        context = operation_context(plan, "second")

        self.assertEqual(context["contract"]["node_id"], "second")
        self.assertEqual(
            [item["node_id"] for item in context["predecessor_contracts"]],
            ["first"],
        )
        self.assertEqual(
            [item["node_id"] for item in context["successor_contracts"]],
            ["third"],
        )
        self.assertNotIn("first", str(context["successor_contracts"]))


class DeterministicDagTests(unittest.TestCase):
    @staticmethod
    def plan() -> WorkflowPlan:
        def data(node_id: str, data_path: str = "") -> PlanNode:
            return PlanNode(
                id=node_id,
                kind="data",
                description=node_id,
                data_path=data_path,
                implementation="generated",
            )

        def operation(
            node_id: str, literals: dict[str, object] | None = None
        ) -> PlanNode:
            return PlanNode(
                id=node_id,
                kind="operation",
                description=node_id,
                implementation="generated",
                literal_arguments=literals or {},
                generation_reason="No registered operation matches.",
            )

        return WorkflowPlan(
            rationale="fully executable test DAG",
            nodes=[
                data("source", "source.geojson"),
                operation("retrieve", {"query": "parks"}),
                data("features"),
                operation("summarize"),
                data("summary"),
                operation("write_manifest"),
                data("manifest", "llm_geo_result.json"),
            ],
            edges=[
                PlanEdge(source="source", target="retrieve"),
                PlanEdge(source="retrieve", target="features"),
                PlanEdge(source="features", target="summarize"),
                PlanEdge(source="summarize", target="summary"),
                PlanEdge(source="summary", target="write_manifest"),
                PlanEdge(source="write_manifest", target="manifest"),
            ],
        )

    def test_main_is_rendered_directly_from_dag(self) -> None:
        main_code = render_main(self.plan())

        self.assertIn("def main():", main_code)
        self.assertIn("source = gpd.read_file('source.geojson')", main_code)
        self.assertIn("features = retrieve(source, query='parks')", main_code)
        self.assertIn("summary = summarize(features)", main_code)
        self.assertIn("manifest = write_manifest(summary)", main_code)
        self.assertTrue(main_code.rstrip().endswith("main()"))

    def test_executable_plan_requires_explicit_manifest_operation(self) -> None:
        plan = self.plan()
        self.assertEqual(
            validate_workflow_plan(
                plan, [], [], require_result_manifest=True
            ),
            [],
        )
        plan.nodes[-1].data_path = "summary.json"

        issues = validate_workflow_plan(
            plan, [], [], require_result_manifest=True
        )

        self.assertTrue(any("llm_geo_result.json" in issue for issue in issues))

    def test_atomic_operation_code_checks_signature_and_return_contract(self) -> None:
        contract = {
            "node_id": "summarize",
            "inputs": ["features"],
            "literal_arguments": {"limit": 10},
            "outputs": ["summary"],
        }
        valid_code = (
            "def summarize(features, limit):\n"
            "    summary = features.head(limit)\n"
            "    return summary\n"
        )
        self.assertEqual(validate_operation_code(valid_code, contract), [])

        issues = validate_operation_code(
            "def summarize(features):\n    return features\n", contract
        )

        self.assertTrue(any("parameters must be exactly" in issue for issue in issues))
        self.assertTrue(any("output variables exactly" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
