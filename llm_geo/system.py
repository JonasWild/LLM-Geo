"""Complete checkpointable LLM-GEO LangGraph workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import textwrap
import time
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
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
    ask_structured_with_tool_results,
    build_review_prompt,
    create_structured_agent,
    review_code,
)
from llm_geo.tools.code_execution import execute_code, snapshot_files
from llm_geo.tools.data_inspection import inspect_source, to_json, to_toon
from llm_geo.tools.data_retrieval import (
    provider_tool_instructions,
    validate_provider_results,
)
from llm_geo.tools.mermaid_diagnostics import (
    execution_event,
    write_execution_graph_artifacts,
    write_system_graph_artifacts,
)
from llm_geo.tools.workflow_graph import (
    operation_contract,
    plan_to_graph,
    validate_workflow_plan,
    write_graph_artifacts,
)
from llm_geo.utils.models import (
    CodeArtifact,
    DataSource,
    LLMGeoState,
    RetrievalDecision,
    ResultValidation,
    ReviewDecision,
    WorkflowPlan,
    WorkflowStep,
)
from llm_geo.utils.prompts import GIS_RULES, save_prompt
from llm_geo.utils.timing import time_node


def create_llm_geo_graph(
    model: BaseChatModel,
    checkpointer: Any | None = None,
    retrieval_tools: Sequence[BaseTool] = (),
    registered_operations: Sequence[RegisteredOperation] = (),
) -> CompiledStateGraph:
    """Create the complete, staged LLM-GEO production graph."""
    provider_tools = list(retrieval_tools)
    trusted_operations = tuple(registered_operations)
    trusted_by_id = {operation.id: operation for operation in trusted_operations}
    retriever = create_structured_agent(
        model,
        "You are the LLM-GEO data retrieval coordinator. Use only the registered "
        "provider tools to retrieve data. Never invent URLs, paths, providers, or "
        "data. After the tools respond, select only locations they returned.",
        RetrievalDecision,
        tools=provider_tools,
    )
    planner = create_structured_agent(
        model,
        "You are the LLM-GEO workflow planner. Produce a concise alternating "
        "data/operation DAG. " + GIS_RULES,
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
    direct_coder = create_structured_agent(
        model,
        "Write one complete robust GIS program for the task, including saved "
        "outputs and a result manifest. " + GIS_RULES,
        CodeArtifact,
    )

    def traced_node(name: str, node: Any) -> Any:
        """Record one checkpointed event around a top-level workflow node."""
        timed_node = time_node(node)

        def invoke(state: LLMGeoState) -> dict[str, Any]:
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            trace = state.get("execution_trace", [])
            try:
                update = asyncio.run(timed_node(state))
            except Exception as error:
                event = execution_event(
                    trace,
                    name,
                    "exception",
                    started_at=started_at,
                    duration_seconds=time.perf_counter() - started,
                    exception_type=type(error).__name__,
                )
                write_execution_graph_artifacts(
                    [*trace, event], Path(state["save_dir"])
                )
                raise
            event = execution_event(
                trace,
                name,
                str(update.get("status", state.get("status", "unknown"))),
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
            )
            return {**update, "execution_trace": [event]}

        return invoke

    def retrieve_sources(state: LLMGeoState) -> dict[str, Any]:
        if not provider_tools:
            return {
                "retrieval_error": "No data retrieval providers are configured.",
                "status": "retrieval_failed",
            }
        data_directory = Path(state["save_dir"]) / "data"
        data_directory.mkdir(parents=True, exist_ok=True)
        get_logger().info("Retrieving data | providers=%d", len(provider_tools))
        prompt = textwrap.dedent(f"""
        TASK:
        {state["task"]}

        {provider_tool_instructions(data_directory)}

        Call the applicable provider tools. Return only the local locations returned
        by those tool calls, in `selected_locations`.
        """)
        prompt_path = save_prompt(
            state["save_dir"],
            stage="retrieve",
            agent="retriever",
            subject="sources",
            prompt=prompt,
        )
        get_logger().info("Retriever prompt saved | path=%s", prompt_path)
        try:
            decision, raw_results = ask_structured_with_tool_results(retriever, prompt)
            if not isinstance(decision, RetrievalDecision):
                raise TypeError("Retriever returned an unexpected response type")
            retrieved = validate_provider_results(raw_results, data_directory)
            sources_by_location = {source.location: source for source in retrieved}
            if len(decision.selected_locations) != len(
                set(decision.selected_locations)
            ):
                raise ValueError("Retriever selected the same location more than once.")
            unknown = set(decision.selected_locations) - set(sources_by_location)
            if unknown:
                raise ValueError(
                    "Retriever selected locations not returned by a provider: "
                    + ", ".join(sorted(unknown))
                )
            sources = [
                sources_by_location[path] for path in decision.selected_locations
            ]
            get_logger().info("Data retrieved | sources=%d", len(sources))
            return {
                "data_sources": [source.model_dump(mode="json") for source in sources],
                "status": "sources_retrieved",
            }
        except Exception as error:
            traceback.print_exception(error)
            message = f"{type(error).__name__}: {error}"
            get_logger().warning("Data retrieval failed | reason=%s", message)
            return {"retrieval_error": message, "status": "retrieval_failed"}

    def inspect_sources(state: LLMGeoState) -> dict[str, Any]:
        sources = [DataSource.model_validate(item) for item in state["data_sources"]]
        get_logger().info("Inspecting data | sources=%d", len(sources))
        inspected = [inspect_source(source) for source in sources]
        failures = sum(source.inspection_error is not None for source in inspected)
        if failures:
            get_logger().warning(
                "Data inspection completed with unavailable metadata | failures=%d",
                failures,
            )
        else:
            get_logger().info("Data inspection complete")
        return {
            "data_sources": [source.model_dump(mode="json") for source in inspected],
            "status": "sources_inspected",
        }

    def plan_workflow(state: LLMGeoState) -> dict[str, Any]:
        attempt = state.get("plan_attempts", 0) + 1
        get_logger().info(
            "Planning workflow | attempt=%d/%d",
            attempt,
            state.get("max_plan_attempts", 3),
        )
        prompt = (
            f"TASK:\n{state['task']}\n\nINSPECTED SOURCES:\n"
            f"{to_toon(state['data_sources'])}\n\nPREVIOUS PLAN ISSUES TO CORRECT:\n"
            f"{to_toon(state.get('plan_issues', []))}\n\nREGISTERED OPERATIONS:\n"
            f"{to_toon([operation.catalog_entry() for operation in trusted_operations])}\n\n"
            "Return all input, intermediate, and final data nodes. Each edge must "
            "alternate data and operation. Copy each system-retrieved GeoJSON source "
            "location exactly into one source data node's data_path. Use GeoPandas to "
            "load source nodes. Data nodes must leave implementation as 'generated' "
            "and registered_operation_id as null. Only operation nodes select an "
            "implementation: for a matching registered operation, set it to "
            "'registered' and set registered_operation_id exactly to the catalog ID. "
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
        sources = [DataSource.model_validate(item) for item in state["data_sources"]]
        issues = validate_workflow_plan(plan, sources, trusted_operations)
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

    def inspection_route(state: LLMGeoState) -> str:
        return "direct" if state.get("direct_mode") else "plan"

    def retrieval_route(state: LLMGeoState) -> str:
        return "failed" if state.get("retrieval_error") else "inspect"

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
                arguments = ", ".join(contract["inputs"])
                bridge_code = (
                    f"from {registered.module} import {registered.name}\n\n"
                    f"def {operation_id}({arguments}):\n"
                    f"    return {registered.name}({arguments})"
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
                f"TASK:\n{state['task']}\n\nSOURCES:\n"
                f"{to_toon(state['data_sources'])}\n\nPLAN:\n{to_toon(state['plan'])}"
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
            reviewed_code, review_issues = review_code(
                reviewer, artifact.code, requirements, prompt=reviewer_prompt
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

        DATA SOURCES:
        {to_toon(state["data_sources"])}

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
        reviewed_code, _ = review_code(
            reviewer, artifact.code, requirements, prompt=reviewer_prompt
        )
        path = Path(state["save_dir"]) / "code" / "solution.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reviewed_code, encoding="utf-8")
        get_logger().info("Program assembled and reviewed | path=%s", path)
        return {
            "assembled_code": reviewed_code,
            "artifacts": state.get("artifacts", []) + [str(path)],
            "status": "program_assembled",
        }

    def generate_direct_program(state: LLMGeoState) -> dict[str, Any]:
        get_logger().info("Generating direct solution | graph decomposition=skipped")
        requirements = f"""
        TASK:
        {state["task"]}

        INSPECTED SOURCES:
        {to_toon(state["data_sources"])}

        Return a complete executable program. Print important results, save requested
        maps/charts, and write llm_geo_result.json with a summary, serializable result
        values, and artifact paths. The current working directory is the run's results
        directory: write every generated artifact there with a relative path. Treat
        the supplied source paths as read-only and never derive output paths from them.
        """
        requirements = textwrap.dedent(requirements)
        direct_prompt_path = save_prompt(
            state["save_dir"],
            stage="direct",
            agent="coder",
            subject="program",
            prompt=requirements,
        )
        get_logger().info("Direct coder prompt saved | path=%s", direct_prompt_path)
        artifact = ask_structured(direct_coder, requirements)
        if not isinstance(artifact, CodeArtifact):
            raise TypeError("Direct coder returned an unexpected response type")
        reviewer_prompt = build_review_prompt(artifact.code, requirements)
        reviewer_prompt_path = save_prompt(
            state["save_dir"],
            stage="direct",
            agent="reviewer",
            subject="program",
            prompt=reviewer_prompt,
        )
        get_logger().info("Reviewer prompt saved | path=%s", reviewer_prompt_path)
        reviewed_code, _ = review_code(
            reviewer, artifact.code, requirements, prompt=reviewer_prompt
        )
        return {
            "assembled_code": reviewed_code,
            "operations": [],
            "status": "direct_program_generated",
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
            execution = execute_code(state["assembled_code"], Path(state["save_dir"]))
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

        DATA SOURCES:
        {to_toon(state["data_sources"])}

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
        write_execution_graph_artifacts(trace, save_dir)
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
        write_execution_graph_artifacts(trace, save_dir)
        state_path = save_dir / f"{state['task_name']}.state.json"
        persisted = {
            key: value for key, value in state.items() if key != "assembled_code"
        }
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
    graph.add_node(
        "retrieve_sources", traced_node("retrieve_sources", retrieve_sources)
    )
    graph.add_node("inspect_sources", traced_node("inspect_sources", inspect_sources))
    graph.add_node("plan_workflow", traced_node("plan_workflow", plan_workflow))
    graph.add_node("validate_plan", traced_node("validate_plan", validate_plan))
    graph.add_node(
        "generate_operations",
        traced_node("generate_operations", generate_operations),
    )
    graph.add_node(
        "generate_direct_program",
        traced_node("generate_direct_program", generate_direct_program),
    )
    graph.add_node(
        "assemble_program", traced_node("assemble_program", assemble_program)
    )
    graph.add_node("execute_program", traced_node("execute_program", execute_program))
    graph.add_node("debug_program", traced_node("debug_program", debug_program))
    graph.add_node("validate_result", traced_node("validate_result", validate_result))
    graph.add_node("finalize_success", finalize_success)
    graph.add_node("finalize_failure", finalize_failure)

    graph.add_edge(START, "retrieve_sources")
    graph.add_conditional_edges(
        "retrieve_sources",
        retrieval_route,
        {"inspect": "inspect_sources", "failed": "finalize_failure"},
    )
    graph.add_conditional_edges(
        "inspect_sources",
        inspection_route,
        {"plan": "plan_workflow", "direct": "generate_direct_program"},
    )
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
    graph.add_edge("generate_direct_program", "execute_program")
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
    retrieval_tools: Sequence[BaseTool] = (),
    registered_operations: Sequence[RegisteredOperation] = (),
    output_root: str | Path = "output",
    direct_mode: bool = False,
    allow_code_execution: bool = True,
    max_plan_attempts: int = 3,
    max_execution_attempts: int = 10,
    log_level: int = logging.INFO,
    log_http: bool = True,
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
        "Workflow started | task=%s | mode=%s | providers=%d | output=%s",
        safe_task_name,
        "direct" if direct_mode else "graph",
        len(retrieval_tools),
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
    graph = create_llm_geo_graph(
        model,
        checkpointer=SqliteSaver(connection),
        retrieval_tools=retrieval_tools,
        registered_operations=registered_operations,
    )
    system_artifacts = write_system_graph_artifacts(graph, destination)
    initial: LLMGeoState = {
        "task": task,
        "task_name": safe_task_name,
        "save_dir": str(destination),
        "direct_mode": direct_mode,
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
        return graph.invoke(initial, config=config)
    finally:
        connection.close()
        close_file_logging()


def resume_llm_geo(
    model: BaseChatModel,
    run_dir: str | Path,
    *,
    retrieval_tools: Sequence[BaseTool] = (),
    registered_operations: Sequence[RegisteredOperation] = (),
    log_level: int = logging.INFO,
    log_http: bool = True,
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
        "Resuming workflow | task=%s | checkpoint=%s", task_name, checkpoint_path
    )
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    graph = create_llm_geo_graph(
        model,
        checkpointer=SqliteSaver(connection),
        retrieval_tools=retrieval_tools,
        registered_operations=registered_operations,
    )
    write_system_graph_artifacts(graph, destination)
    config = {
        "configurable": {"thread_id": task_name},
        "recursion_limit": 100,
    }
    try:
        result = graph.invoke(None, config=config)
        get_logger().info("Resume finished | status=%s", result.get("status"))
        return result
    finally:
        connection.close()
        close_file_logging()
