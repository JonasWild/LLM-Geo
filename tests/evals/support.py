"""Small deterministic DeepEval adapter and metrics."""

from __future__ import annotations

import json
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from llm_geo.tools.workflow_graph import validate_workflow_plan
from llm_geo.utils.models import DataSource, WorkflowPlan


def workflow_test_case(task: str, state: dict[str, Any]) -> LLMTestCase:
    """Represent a completed workflow state as a DeepEval test case."""
    return LLMTestCase(
        input=task,
        actual_output=json.dumps(state, sort_keys=True),
    )


def _state(test_case: LLMTestCase) -> dict[str, Any]:
    if test_case.actual_output is None:
        raise ValueError("The workflow test case has no actual output.")
    value = json.loads(test_case.actual_output)
    if not isinstance(value, dict):
        raise ValueError("The workflow state must be a JSON object.")
    return value


class WorkflowCompletionMetric(BaseMetric):
    """Require successful execution and final validation."""

    def __init__(self) -> None:
        self.threshold = 1.0
        self.async_mode = True
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "Workflow completion"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        state = _state(test_case)
        failed_checks = []
        if state.get("status") != "complete":
            failed_checks.append("status is not complete")
        if not state.get("execution", {}).get("success"):
            failed_checks.append("execution did not succeed")
        if not state.get("validation", {}).get("valid"):
            failed_checks.append("result validation did not pass")
        self.score = 0.0 if failed_checks else 1.0
        self.reason = "; ".join(failed_checks) or "Workflow completed and validated."
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self, test_case: LLMTestCase, *args: Any, **kwargs: Any
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


class PlanValidityMetric(BaseMetric):
    """Reuse LLM-GEO's deterministic graph validator inside DeepEval."""

    def __init__(self) -> None:
        self.threshold = 1.0
        self.async_mode = True
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "Workflow plan validity"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        state = _state(test_case)
        plan = WorkflowPlan.model_validate(state["plan"])
        sources = [DataSource.model_validate(item) for item in state["data_sources"]]
        issues = validate_workflow_plan(plan, sources)
        self.score = 0.0 if issues else 1.0
        self.reason = "; ".join(issues) or "Workflow plan is valid."
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self, test_case: LLMTestCase, *args: Any, **kwargs: Any
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold
