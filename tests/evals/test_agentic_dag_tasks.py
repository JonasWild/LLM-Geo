"""Opt-in LLM-judged integration evaluations for complex GIS workflows."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

import main as runtime
from llm_geo.system import run_llm_geo
from tests.evals.dag_scenarios import DAG_TASKS
from tests.evals.support import workflow_test_case


pytestmark = [pytest.mark.eval, pytest.mark.llm, pytest.mark.integration]

DAG_JUDGE = GEval(
    name="Agentic GIS DAG quality",
    criteria="""
Evaluate whether the workflow state demonstrates a coherent, task-complete GIS
solution. Judge whether the plan decomposes the request into appropriate
retrieval, transformation, analysis, and output steps; dependencies form a
logical medium-to-high complexity DAG; provider and tool choices fit the
requested capabilities; executed results are consistent with plan and tool
evidence; validation and artifacts plausibly satisfy every requested output;
and the workflow avoids unsupported claims, invented data, and unnecessary
steps. Score low if success is asserted without supporting execution, tool,
validation, and artifact evidence.
""".strip(),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
)


def _evals_enabled() -> bool:
    return os.getenv("LLM_GEO_RUN_DAG_EVALS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.parametrize("scenario_id,task", DAG_TASKS, ids=[item[0] for item in DAG_TASKS])
def test_agentic_dag_task(
    tmp_path: Path,
    scenario_id: str,
    task: str,
) -> None:
    if not _evals_enabled():
        pytest.skip("Set LLM_GEO_RUN_DAG_EVALS=1 to run live agentic DAG evals")

    model = runtime.initialize_model()
    state = run_llm_geo(
        model=model,
        task=task,
        task_name=scenario_id,
        registered_operations=runtime.REGISTERED_OPERATIONS,
        operation_retriever=runtime.initialize_operation_retriever(),
        operation_retrieval_limit=runtime.OPERATION_RETRIEVAL_LIMIT,
        output_root=tmp_path,
        allow_code_execution=runtime.ALLOW_CODE_EXECUTION,
        max_plan_attempts=runtime.MAX_PLAN_ATTEMPTS,
        max_execution_attempts=runtime.MAX_EXECUTION_ATTEMPTS,
        log_level=runtime.LOG_LEVEL,
        log_http=runtime.LOG_HTTP,
        generate_mermaid=runtime.GENERATE_MERMAID,
        slow_step_seconds=runtime.SLOW_STEP_SECONDS,
    )

    assert_test(workflow_test_case(task, state), [DAG_JUDGE])
