"""Complete checkpointable LLM-GEO LangGraph workflow."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import textwrap
import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from llm_geo.middleware.logging import (
    close_file_logging,
    configure_logging,
    get_logger,
)
from llm_geo.operations.registry import RegisteredOperation
from llm_geo.subagents.runtime import (
    ask_structured,
    build_review_prompt,
    create_structured_agent,
    review_code,
)
from llm_geo.tools.code_execution import execute_code, snapshot_files
from llm_geo.tools.data_inspection import to_json, to_toon
from llm_geo.tools.mermaid_diagnostics import (
    execution_event,
    write_execution_graph_artifacts,
    write_system_graph_artifacts,
)
from llm_geo.tools.workflow_graph import (
    operation_contract,
    plan_to_graph,
    registered_operation_bridge,
    validate_workflow_plan,
    write_graph_artifacts,
)
from llm_geo.utils.models import (
    CodeArtifact,
    LLMGeoState,
    ResultValidation,
    ReviewDecision,
    WorkflowPlan,
    WorkflowStep,
)
from llm_geo.utils.prompts import GIS_RULES, save_prompt


@contextmanager
def timed_step(
    name: str,
    threshold_seconds: float,
    **fields: Any,
):
    """Log a completed or failed operation only when it crosses the threshold."""
    started = time.perf_counter()
    error: BaseException | None = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        duration = time.perf_counter() - started
        if duration >= threshold_seconds:
            details = "".join(f" | {key}={value}" for key, value in fields.items())
            if error is None:
                get_logger().warning(
                    "Slow step | step=%s%s | duration=%.3fs",
                    name,
                    details,
                    duration,
                )
            else:
                get_logger().warning(
                    "Slow step failed | step=%s%s | type=%s | duration=%.3fs",
                    name,
                    details,
                    type(error).__name__,
                    duration,
                )


def create_llm_geo_graph(
    model: BaseChatModel,
    checkpointer: Any | None = None,
    registered_operations: Sequence[RegisteredOperation] = (),
    generate_mermaid: bool = True,
    slow_step_seconds: float = 10.0,
) -> CompiledStateGraph:
    """Create the complete, staged LLM-GEO production graph."""
    trusted_operations = tuple(registered_operations)
    trusted_by_id = {operation.id: operation for operation in trusted_operations}
    planner = create_structured_agent(
        model,
        "You are the LLM-GEO workflow planner. Produce one concise alternating "
        "data/operation DAG that includes data retrieval. Select trusted registered "
        "operations whenever their contracts match. " + GIS_RULES,
        WorkflowPlan,
    )
    coder = create_structured_agent(
        model,
        "You implement exactly one robust GIS operation function while preserving "
        "its required interface. " + GIS_RULES,
        CodeArtifact,
    )
    reviewer = create_structured_agent(
        model,
        "You are a strict GIS Python reviewer. Pass correct code or return the "
        "complete corrected code. " + GIS_RULES,
        ReviewDecision,
    )
    assembler = create_structured_agent(
        model,
        "You assemble reviewed GIS functions into one complete executable Python "
        "program. " + GIS_RULES,
        CodeArtifact,
    )
    debugger = create_structured_agent(
        model,
        "Repair the complete Python program from its traceback or validation "
        "failures. Preserve intended interfaces. " + GIS_RULES,
        CodeArtifact,
    )
    validator = create_structured_agent(
        model,
        "Validate whether an executed GIS program actually answered the task. "
        "Reject plausible-looking but unverified results. " + GIS_RULES,
        ResultValidation,
    )
    def traced_node(name: str, node: Any) -> Any:
        """Log, time, and checkpoint one top-level workflow node."""

        def invoke(state: LLMGeoState) -> dict[str, Any]:
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            trace = state.get("execution_trace", [])
            occurrence = sum(event.get("node") == name for event in trace) + 1
            logger = get_logger()
            logger.info("%s start | occurrence=%d", name, occurrence)
            try:
                update = node(state)
            except Exception as error:
                duration = time.perf_counter() - started
                event = execution_event(
                    trace,
                    name,
                    "exception",
                    started_at=started_at,
                    duration_seconds=duration,
                    exception_type=type(error).__name__,
                )
                logger.exception(
                    "%s failed | type=%s | duration=%.3fs | reason=%s",
                    name,
                    type(error).__name__,
                    duration,
                    error,
                )
                if generate_mermaid:
                    with timed_step(
                        "mermaid.execution", slow_step_seconds, outcome="exception"
                    ):
                        write_execution_graph_artifacts(
                            [*trace, event], Path(state["save_dir"])
                        )
                raise
            duration = time.perf_counter() - started
            status = str(update.get("status", state.get("status", "unknown")))
            event = execution_event(
                trace,
                name,
                status,
                started_at=started_at,
                duration_seconds=duration,
            )
            logger.info(
                "%s done | status=%s | duration=%.3fs",
                name,
                status,
                duration,
            )
            return {**update, "execution_trace": [event]}

        return invoke

    def plan_workflow(state: LLMGeoState) -> dict[str, Any]:
        attempt = state.get("plan_attempts", 0) + 1
        get_logger().info(
            "Planning workflow | attempt=%d/%d",
            attempt,
            state.get("max_plan_attempts", 3),
        )
        prompt = (
            f"TASK:\n{state['task']}\n\nPREVIOUS PLAN ISSUES TO CORRECT:\n"
            f"{to_toon(state.get('plan_issues', []))}\n\nREGISTERED OPERATIONS:\n"
            f"{to_toon([operation.catalog_entry() for operation in trusted_operations])}\n\n"
            "Return one DAG containing retrieval, transformation, and output. Each "
            "edge must alternate data and operation. Retrieval must use matching "
            "registered operations; their outputs are EPSG:4326 GeoDataFrames also "
            "persisted as GeoJSON. Put task-derived scalar/configuration values in "
            "the operation's literal_arguments. Incoming data edges bind in declared "
            "parameter order to parameters not supplied as literals. Data nodes must "
            "leave implementation as 'generated' "
            "and registered_operation_id as null. Only operation nodes select an "
            "implementation: for a matching registered operation, set it to "
            "'registered' and set registered_operation_id exactly to the catalog ID. "
            "A registered retrieval operation may have no incoming data edge when "
            "literal arguments and defaults satisfy its parameters. "
            "For a generated operation, use implementation 'generated' and omit "
            "registered_operation_id."
        )
        prompt_path = save_prompt(
            state["save_dir"],
            stage="plan",
            agent="planner",
            subject="workflow",
            prompt=prompt,
        )
        get_logger().info("Planner prompt saved | path=%s", prompt_path)
        with timed_step("planner.call", slow_step_seconds):
            result = ask_structured(planner, prompt)
        if not isinstance(result, WorkflowPlan):
            raise TypeError("Planner returned an unexpected response type")
        return {
            "plan": result.model_dump(mode="json"),
            "plan_attempts": attempt,
            "artifacts": state.get("artifacts", []) + [str(prompt_path)],
            "status": "plan_created",
        }

    def validate_plan(state: LLMGeoState) -> dict[str, Any]:
        plan = WorkflowPlan.model_validate(state["plan"])
        issues = validate_workflow_plan(plan, [], trusted_operations)
        if issues:
            get_logger().warning(
                "Workflow plan rejected | issues=%d | %s",
                len(issues),
                "; ".join(issues[:3]),
            )
            return {"plan_issues": issues, "status": "plan_invalid"}
        operations = sum(node.kind == "operation" for node in plan.nodes)
        get_logger().info(
            "Workflow plan valid | nodes=%d | operations=%d | edges=%d",
            len(plan.nodes),
            operations,
            len(plan.edges),
        )
        artifacts = write_graph_artifacts(
            plan, Path(state["save_dir"]), state["task_name"]
        )
        get_logger().info("Workflow graphs saved | formats=GraphML,PNG,HTML")
        return {"plan_issues": [], "artifacts": artifacts, "status": "plan_valid"}

    def plan_route(state: LLMGeoState) -> str:
        if not state.get("plan_issues"):
            return "operations"
        if state.get("plan_attempts", 0) < state.get("max_plan_attempts", 3):
            return "replan"
        return "failed"

    def generate_operations(state: LLMGeoState) -> dict[str, Any]:
        plan = WorkflowPlan.model_validate(state["plan"])
        graph = plan_to_graph(plan)
        node_map = {node.id: node for node in plan.nodes}
        operation_ids = [
            node_id
            for node_id in nx.topological_sort(graph)
            if node_map[node_id].kind == "operation"
        ]
        completed: dict[str, dict[str, Any]] = {}
        operations: list[dict[str, Any]] = []
        for index, operation_id in enumerate(operation_ids, start=1):
            get_logger().info(
                "Generating operation | %d/%d | node=%s",
                index,
                len(operation_ids),
                operation_id,
            )
            contract = operation_contract(plan, operation_id)
            selected_id = node_map[operation_id].registered_operation_id
            if node_map[operation_id].implementation == "registered":
                registered = trusted_by_id[selected_id or ""]
                bridge_code = registered_operation_bridge(
                    plan, operation_id, registered
                )
                step = WorkflowStep(
                    node_id=operation_id,
                    description=contract["description"],
                    inputs=contract["inputs"],
                    outputs=contract["outputs"],
                    code=bridge_code,
                    registered_operation_id=registered.id,
                ).model_dump(mode="json")
                completed[operation_id] = step
                operations.append(step)
                get_logger().info(
                    "Using registered operation | node=%s | operation=%s",
                    operation_id,
                    registered.id,
                )
                continue
            ancestor_ids = {
                node_id
                for node_id in nx.ancestors(graph, operation_id)
                if node_map[node_id].kind == "operation"
            }
            ancestor_code = "\n\n".join(
                completed[node_id]["code"]
                for node_id in operation_ids
                if node_id in ancestor_ids
            )
            descendant_contracts = [
                operation_contract(plan, node_id)
                for node_id in operation_ids
                if node_id in nx.descendants(graph, operation_id)
            ]
            requirements = (
                f"TASK:\n{state['task']}\n\nPLAN:\n{to_toon(state['plan'])}"
                f"\n\nCONTRACT:\n{to_toon(contract)}\n\nANCESTOR CODE:\n"
                f"{ancestor_code or '(none)'}\n\nDESCENDANT CONTRACTS:\n"
                f"{to_toon(descendant_contracts)}\n\nThe returned code must start "
                f"with `{contract['signature']}` and end by returning exactly "
                f"{contract['outputs']}. Do not call the function."
            )
            coder_prompt_path = save_prompt(
                state["save_dir"],
                stage="ops",
                agent="coder",
                subject=operation_id,
                prompt=requirements,
            )
            get_logger().info("Coder prompt saved | path=%s", coder_prompt_path)
            with timed_step(
                "coder.call", slow_step_seconds, operation=operation_id
            ):
                artifact = ask_structured(coder, requirements)
            if not isinstance(artifact, CodeArtifact):
                raise TypeError("Coder returned an unexpected response type")
            reviewer_prompt = build_review_prompt(artifact.code, requirements)
            reviewer_prompt_path = save_prompt(
                state["save_dir"],
                stage="ops",
                agent="reviewer",
                subject=operation_id,
                prompt=reviewer_prompt,
            )
            get_logger().info("Reviewer prompt saved | path=%s", reviewer_prompt_path)
            with timed_step(
                "reviewer.call", slow_step_seconds, operation=operation_id
            ):
                reviewed_code, review_issues = review_code(
                    reviewer, artifact.code, requirements, prompt=reviewer_prompt
                )
            get_logger().info(
                "Operation reviewed | node=%s | issues=%d",
                operation_id,
                len(review_issues),
            )
            step = WorkflowStep(
                node_id=operation_id,
                description=contract["description"],
                inputs=contract["inputs"],
                outputs=contract["outputs"],
                code=reviewed_code,
                review_issues=review_issues,
            ).model_dump(mode="json")
            completed[operation_id] = step
            operations.append(step)
        get_logger().info("Operation generation complete | count=%d", len(operations))
        return {"operations": operations, "status": "operations_generated"}

    def assemble_program(state: LLMGeoState) -> dict[str, Any]:
        get_logger().info("Assembling complete program")
        function_code = "\n\n".join(item["code"] for item in state["operations"])
        requirements = f"""
        TASK:
        {state["task"]}

        WORKFLOW PLAN:
        {to_toon(state["plan"])}

        REVIEWED FUNCTIONS:
        {function_code}

        Return a complete program containing all required imports, the reviewed
        functions unchanged, and an assemble_solution() function that calls them in
        dependency order. Call assemble_solution() at the end. Print important results.
        Save requested maps/charts and write llm_geo_result.json in the current
        working directory, which is the run's results directory. Do not use an
        if __name__ guard. Functions that import from `llm_geo.operations` are trusted
        registered operations: retain those import statements and function calls
        exactly, and do not rewrite their implementations.
        """
        requirements = textwrap.dedent(requirements)
        assembler_prompt_path = save_prompt(
            state["save_dir"],
            stage="assemble",
            agent="assembler",
            subject="program",
            prompt=requirements,
        )
        get_logger().info("Assembler prompt saved | path=%s", assembler_prompt_path)
        with timed_step("assembler.call", slow_step_seconds):
            artifact = ask_structured(assembler, requirements)
        if not isinstance(artifact, CodeArtifact):
            raise TypeError("Assembler returned an unexpected response type")
        reviewer_prompt = build_review_prompt(artifact.code, requirements)
        reviewer_prompt_path = save_prompt(
            state["save_dir"],
            stage="assemble",
            agent="reviewer",
            subject="program",
            prompt=reviewer_prompt,
        )
        get_logger().info("Reviewer prompt saved | path=%s", reviewer_prompt_path)
        with timed_step("reviewer.call", slow_step_seconds, subject="assembly"):
            reviewed_code, review_issues = review_code(
                reviewer, artifact.code, requirements, prompt=reviewer_prompt
            )
        get_logger().info("Assembly reviewed | issues=%d", len(review_issues))
        path = Path(state["save_dir"]) / "code" / "solution.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reviewed_code, encoding="utf-8")
        get_logger().info("Program assembled and reviewed | path=%s", path)
        return {
            "assembled_code": reviewed_code,
            "artifacts": state.get("artifacts", []) + [str(path)],
            "status": "program_assembled",
        }

    def execute_program(state: LLMGeoState) -> dict[str, Any]:
        attempt = state.get("execution_attempts", 0) + 1
        get_logger().info(
            "Executing program | attempt=%d/%d",
            attempt,
            state.get("max_execution_attempts", 10),
        )
        if not state.get("allow_code_execution", True):
            execution = {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": "Code execution is disabled.",
                "new_files": [],
            }
            get_logger().warning("Execution skipped | code execution is disabled")
        else:
            with timed_step("code.execute", slow_step_seconds, attempt=attempt):
                execution = execute_code(
                    state["assembled_code"], Path(state["save_dir"])
                )
            get_logger().info(
                "Code run result | exit=%s | files=%d",
                execution.get("returncode"),
                len(execution.get("new_files", [])),
            )
        return {
            "execution": execution,
            "execution_attempts": attempt,
            "status": (
                "execution_succeeded" if execution["success"] else "execution_failed"
            ),
        }

    def execution_route(state: LLMGeoState) -> str:
        if state["execution"]["success"]:
            return "validate"
        if not state.get("allow_code_execution", True):
            return "failed"
        if state.get("execution_attempts", 0) < state.get("max_execution_attempts", 10):
            return "debug"
        return "failed"

    def debug_program(state: LLMGeoState) -> dict[str, Any]:
        get_logger().info(
            "Repairing program | completed_attempts=%d",
            state.get("execution_attempts", 0),
        )
        prompt = f"""
        TASK:
        {state["task"]}

        FAILURE OR VALIDATION ISSUES:
        {to_toon(state.get("execution", {}))}
        {to_toon(state.get("validation", {}))}

        COMPLETE PROGRAM:
        {state["assembled_code"]}

        Return the entire corrected program, not a patch.
        """
        prompt = textwrap.dedent(prompt)
        debugger_prompt_path = save_prompt(
            state["save_dir"],
            stage="debug",
            agent="debugger",
            subject="program",
            prompt=prompt,
        )
        get_logger().info("Debugger prompt saved | path=%s", debugger_prompt_path)
        with timed_step("debugger.call", slow_step_seconds):
            artifact = ask_structured(debugger, prompt)
        if not isinstance(artifact, CodeArtifact):
            raise TypeError("Debugger returned an unexpected response type")
        get_logger().info("Program repair generated | retrying execution")
        return {"assembled_code": artifact.code, "status": "program_repaired"}

    def validate_result(state: LLMGeoState) -> dict[str, Any]:
        get_logger().info("Validating final result")
        save_dir = Path(state["save_dir"])
        manifest_path = save_dir / "results" / "llm_geo_result.json"
        deterministic_issues: list[str] = []
        manifest: Any = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as error:
                deterministic_issues.append(f"Invalid result manifest: {error}")
        else:
            deterministic_issues.append("llm_geo_result.json was not created.")
        prompt = f"""
        TASK:
        {state["task"]}

        PLAN:
        {to_toon(state.get("plan", {}))}

        EXECUTION:
        {to_toon(state["execution"])}

        RESULT MANIFEST:
        {to_toon(manifest)}

        DETERMINISTIC ISSUES:
        {to_toon(deterministic_issues)}

        Validate task completion, numerical/geospatial plausibility, required outputs,
        and evidence. If invalid and repairable, return the entire corrected program in
        corrected_code.
        """
        prompt = textwrap.dedent(prompt)
        validator_prompt_path = save_prompt(
            state["save_dir"],
            stage="validate",
            agent="validator",
            subject="result",
            prompt=prompt,
        )
        get_logger().info("Validator prompt saved | path=%s", validator_prompt_path)
        with timed_step("validator.call", slow_step_seconds):
            decision = ask_structured(validator, prompt)
        if not isinstance(decision, ResultValidation):
            raise TypeError("Validator returned an unexpected response type")
        issues = deterministic_issues + decision.issues
        valid = not deterministic_issues and decision.valid
        update: dict[str, Any] = {
            "validation": {
                "valid": valid,
                "issues": issues,
                "has_corrected_code": bool(decision.corrected_code),
            },
            "status": "validated" if valid else "validation_failed",
        }
        if valid:
            get_logger().info("Result validation passed")
        else:
            get_logger().warning(
                "Result validation failed | issues=%d | %s",
                len(issues),
                "; ".join(issues[:3]),
            )
        if not valid and decision.corrected_code:
            update["assembled_code"] = decision.corrected_code
            get_logger().info("Validator supplied corrected program | re-executing")
        return update

    def validation_route(state: LLMGeoState) -> str:
        if state["validation"]["valid"]:
            return "complete"
        if state.get("execution_attempts", 0) < state.get("max_execution_attempts", 10):
            return (
                "execute" if state["validation"].get("has_corrected_code") else "debug"
            )
        return "failed"

    def finalize_success(state: LLMGeoState) -> dict[str, Any]:
        save_dir = Path(state["save_dir"])
        trace = [
            *state.get("execution_trace", []),
            execution_event(
                state.get("execution_trace", []), "finalize_success", "complete"
            ),
        ]
        if generate_mermaid:
            with timed_step("mermaid.execution", slow_step_seconds):
                write_execution_graph_artifacts(trace, save_dir)
        with timed_step("artifacts.snapshot", slow_step_seconds):
            artifact_paths = sorted(snapshot_files(save_dir))
        state_path = save_dir / f"{state['task_name']}.state.json"
        persisted = {
            key: value for key, value in state.items() if key != "assembled_code"
        }
        persisted.update(
            {
                "status": "complete",
                "artifacts": artifact_paths,
                "execution_trace": trace,
            }
        )
        state_path.write_text(to_json(persisted), encoding="utf-8")
        final_artifacts = artifact_paths + [str(state_path)]
        get_logger().info(
            "Workflow complete | attempts=%d | artifacts=%d | output=%s",
            state.get("execution_attempts", 0),
            len(final_artifacts),
            save_dir,
        )
        return {
            "status": "complete",
            "artifacts": final_artifacts,
            "execution_trace": [trace[-1]],
        }

    def finalize_failure(state: LLMGeoState) -> dict[str, Any]:
        if state.get("plan_issues"):
            error = "Workflow planning failed: " + "; ".join(state["plan_issues"])
        else:
            error = state.get("execution", {}).get(
                "stderr", "Workflow failed validation."
            )
        save_dir = Path(state["save_dir"])
        trace = [
            *state.get("execution_trace", []),
            execution_event(
                state.get("execution_trace", []), "finalize_failure", "failed"
            ),
        ]
        if generate_mermaid:
            with timed_step("mermaid.execution", slow_step_seconds):
                write_execution_graph_artifacts(trace, save_dir)
        state_path = save_dir / f"{state['task_name']}.state.json"
        persisted = {
            key: value for key, value in state.items() if key != "assembled_code"
        }
        with timed_step("artifacts.snapshot", slow_step_seconds):
            artifact_paths = sorted(snapshot_files(save_dir))
        persisted.update(
            {
                "status": "failed",
                "error": error,
                "artifacts": artifact_paths,
                "execution_trace": trace,
            }
        )
        state_path.write_text(to_json(persisted), encoding="utf-8")
        get_logger().error("Workflow failed | reason=%s | state=%s", error, state_path)
        return {
            "status": "failed",
            "error": error,
            "artifacts": artifact_paths + [str(state_path)],
            "execution_trace": [trace[-1]],
        }

    graph = StateGraph(LLMGeoState)
    graph.add_node("plan_workflow", traced_node("plan_workflow", plan_workflow))
    graph.add_node("validate_plan", traced_node("validate_plan", validate_plan))
    graph.add_node(
        "generate_operations",
        traced_node("generate_operations", generate_operations),
    )
    graph.add_node(
        "assemble_program", traced_node("assemble_program", assemble_program)
    )
    graph.add_node("execute_program", traced_node("execute_program", execute_program))
    graph.add_node("debug_program", traced_node("debug_program", debug_program))
    graph.add_node("validate_result", traced_node("validate_result", validate_result))
    graph.add_node("finalize_success", finalize_success)
    graph.add_node("finalize_failure", finalize_failure)

    graph.add_edge(START, "plan_workflow")
    graph.add_edge("plan_workflow", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        plan_route,
        {
            "replan": "plan_workflow",
            "operations": "generate_operations",
            "failed": "finalize_failure",
        },
    )
    graph.add_edge("generate_operations", "assemble_program")
    graph.add_edge("assemble_program", "execute_program")
    graph.add_conditional_edges(
        "execute_program",
        execution_route,
        {
            "validate": "validate_result",
            "debug": "debug_program",
            "failed": "finalize_failure",
        },
    )
    graph.add_edge("debug_program", "execute_program")
    graph.add_conditional_edges(
        "validate_result",
        validation_route,
        {
            "complete": "finalize_success",
            "execute": "execute_program",
            "debug": "debug_program",
            "failed": "finalize_failure",
        },
    )
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def run_llm_geo(
    model: BaseChatModel,
    task: str,
    task_name: str,
    *,
    registered_operations: Sequence[RegisteredOperation] = (),
    output_root: str | Path = "output",
    allow_code_execution: bool = True,
    max_plan_attempts: int = 3,
    max_execution_attempts: int = 10,
    log_level: int = logging.INFO,
    log_http: bool = True,
    generate_mermaid: bool = True,
    slow_step_seconds: float = 10.0,
) -> LLMGeoState:
    """Create an isolated run and execute the complete LLM-GEO graph."""
    safe_task_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name.strip()).strip("._")
    if not safe_task_name:
        raise ValueError("task_name must contain at least one letter or number")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(output_root).resolve() / safe_task_name / timestamp
    destination.mkdir(parents=True, exist_ok=True)
    for directory_name in ("prompts", "data", "workflow", "code", "results"):
        (destination / directory_name).mkdir(exist_ok=True)
    configure_logging(log_level, destination / "llm_geo.log", log_http=log_http)
    get_logger().info(
        "Workflow started | task=%s | operations=%d | mermaid=%s | slow=%.3fs | output=%s",
        safe_task_name,
        len(registered_operations),
        "enabled" if generate_mermaid else "disabled",
        slow_step_seconds,
        destination,
    )
    if safe_task_name != task_name:
        get_logger().info(
            "Task name normalized | original=%s | directory=%s",
            task_name,
            safe_task_name,
        )
    checkpoint_path = destination / f"{safe_task_name}.checkpoints.sqlite"
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    with timed_step("graph.compile", slow_step_seconds):
        graph = create_llm_geo_graph(
            model,
            checkpointer=SqliteSaver(connection),
            registered_operations=registered_operations,
            generate_mermaid=generate_mermaid,
            slow_step_seconds=slow_step_seconds,
        )
    system_artifacts: list[str] = []
    if generate_mermaid:
        with timed_step("mermaid.system", slow_step_seconds):
            system_artifacts = write_system_graph_artifacts(graph, destination)
    initial: LLMGeoState = {
        "task": task,
        "task_name": safe_task_name,
        "save_dir": str(destination),
        "allow_code_execution": allow_code_execution,
        "max_plan_attempts": max_plan_attempts,
        "max_execution_attempts": max_execution_attempts,
        "plan_attempts": 0,
        "execution_attempts": 0,
        "operations": [],
        "artifacts": system_artifacts,
        "execution_trace": [],
        "status": "started",
        "checkpoint_thread_id": safe_task_name,
    }
    config = {
        "configurable": {"thread_id": safe_task_name},
        "recursion_limit": 100,
    }
    try:
        with timed_step("workflow.invoke", slow_step_seconds):
            return graph.invoke(initial, config=config)
    finally:
        connection.close()
        close_file_logging()


def resume_llm_geo(
    model: BaseChatModel,
    run_dir: str | Path,
    *,
    registered_operations: Sequence[RegisteredOperation] = (),
    log_level: int = logging.INFO,
    log_http: bool = True,
    generate_mermaid: bool = True,
    slow_step_seconds: float = 10.0,
) -> LLMGeoState:
    """Resume an interrupted run from its durable checkpoint."""
    destination = Path(run_dir).resolve()
    checkpoints = list(destination.glob("*.checkpoints.sqlite"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint database in {destination}, found {len(checkpoints)}"
        )
    checkpoint_path = checkpoints[0]
    task_name = checkpoint_path.name.removesuffix(".checkpoints.sqlite")
    configure_logging(log_level, destination / "llm_geo.log", log_http=log_http)
    get_logger().info(
        "Resuming workflow | task=%s | mermaid=%s | slow=%.3fs | checkpoint=%s",
        task_name,
        "enabled" if generate_mermaid else "disabled",
        slow_step_seconds,
        checkpoint_path,
    )
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    with timed_step("graph.compile", slow_step_seconds):
        graph = create_llm_geo_graph(
            model,
            checkpointer=SqliteSaver(connection),
            registered_operations=registered_operations,
            generate_mermaid=generate_mermaid,
            slow_step_seconds=slow_step_seconds,
        )
    if generate_mermaid:
        with timed_step("mermaid.system", slow_step_seconds):
            write_system_graph_artifacts(graph, destination)
    config = {
        "configurable": {"thread_id": task_name},
        "recursion_limit": 100,
    }
    try:
        with timed_step("workflow.resume", slow_step_seconds):
            result = graph.invoke(None, config=config)
        get_logger().info("Resume finished | status=%s", result.get("status"))
        return result
    finally:
        connection.close()
        close_file_logging()
