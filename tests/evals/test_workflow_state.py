"""Deterministic DeepEval coverage for an LLM-GEO workflow result."""

from __future__ import annotations

from copy import deepcopy

from deepeval import assert_test

from tests.evals.support import (
    PlanValidityMetric,
    WorkflowCompletionMetric,
    workflow_test_case,
)


COMPLETED_STATE = {
    "status": "complete",
    "data_sources": [
        {
            "description": "Fixture places",
            "location": "fixture.geojson",
            "format": "GeoJSON",
            "provider": "fixture_provider",
        }
    ],
    "plan": {
        "rationale": "Render the retrieved places.",
        "nodes": [
            {
                "id": "source",
                "kind": "data",
                "description": "Retrieved places",
                "data_path": "fixture.geojson",
            },
            {
                "id": "render",
                "kind": "operation",
                "description": "Render places to a PNG",
            },
            {
                "id": "result",
                "kind": "data",
                "description": "Rendered map",
            },
        ],
        "edges": [
            {"source": "source", "target": "render"},
            {"source": "render", "target": "result"},
        ],
    },
    "execution": {"success": True, "returncode": 0},
    "validation": {"valid": True, "issues": []},
}


def test_completed_workflow_passes_deepeval() -> None:
    test_case = workflow_test_case("Render the fixture places as a PNG.", COMPLETED_STATE)

    assert_test(
        test_case,
        [WorkflowCompletionMetric(), PlanValidityMetric()],
    )


def test_invalid_plan_is_detected() -> None:
    state = deepcopy(COMPLETED_STATE)
    state["plan"]["edges"] = [
        {"source": "source", "target": "result"},
        {"source": "result", "target": "render"},
    ]
    metric = PlanValidityMetric()

    score = metric.measure(workflow_test_case("Render places.", state))

    assert score == 0.0
    assert not metric.is_successful()
    assert "alternate data and operation" in metric.reason
