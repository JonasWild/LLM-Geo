"""Deep Agents supervisor over the deterministic LLM-GEO workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from llm_geo.operations.registry import RegisteredOperation
from llm_geo.system import run_llm_geo
from llm_geo.tools.data_inspection import from_toon, to_toon


def create_geo_agent(
    model: BaseChatModel,
    registered_operations: Sequence[RegisteredOperation] = (),
    *,
    default_task_name: str | None = None,
    output_root: str | Path = "output",
    allow_code_execution: bool = True,
    max_plan_attempts: int = 3,
    max_execution_attempts: int = 10,
    log_level: int = logging.INFO,
    log_http: bool = True,
    generate_mermaid: bool = True,
    slow_step_seconds: float = 10.0,
) -> CompiledStateGraph:
    """Create a supervisor that delegates complete jobs to LLM-GEO."""

    @tool
    def run_geospatial_analysis(task: str, task_name: str | None = None) -> str:
        """Run the planned, reviewed, executed, and validated geospatial workflow."""
        selected_task_name = default_task_name or task_name
        if not selected_task_name:
            raise ValueError(
                "task_name is required when no default task name is configured"
            )
        result = run_llm_geo(
            model,
            task,
            selected_task_name,
            registered_operations=registered_operations,
            output_root=output_root,
            allow_code_execution=allow_code_execution,
            max_plan_attempts=max_plan_attempts,
            max_execution_attempts=max_execution_attempts,
            log_level=log_level,
            log_http=log_http,
            generate_mermaid=generate_mermaid,
            slow_step_seconds=slow_step_seconds,
        )
        return to_toon(
            {
                "status": result.get("status"),
                "error": result.get("error"),
                "save_dir": result.get("save_dir"),
                "execution": result.get("execution"),
                "validation": result.get("validation"),
                "artifacts": result.get("artifacts", []),
            }
        )

    return create_deep_agent(
        model=model,
        tools=[run_geospatial_analysis],
        system_prompt=(
            "You are the LLM-GEO supervisor. For every requested geospatial analysis, "
            "call run_geospatial_analysis. Data may be retrieved only through the "
            "workflow's registered operations. Never claim execution or validation "
            "unless the workflow tool reports it."
        ),
        backend=StateBackend(),
        checkpointer=InMemorySaver(),
    )


def run_geo_agent(
    agent: CompiledStateGraph,
    task: str,
    task_name: str,
) -> dict[str, Any]:
    """Run one supervised analysis and return its verified workflow summary."""
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Run this geospatial analysis exactly once:\n{task}\n\n"
                        f"Use the configured task name {task_name!r}. Report only the "
                        "status and evidence returned by run_geospatial_analysis."
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": f"geo-supervisor-{task_name}"}},
    )
    tool_messages = [
        message
        for message in response.get("messages", [])
        if isinstance(message, ToolMessage)
        and message.name == "run_geospatial_analysis"
    ]
    if not tool_messages:
        raise RuntimeError("Deep Agent did not call run_geospatial_analysis")
    summary = from_toon(str(tool_messages[-1].content))
    if not isinstance(summary, dict):
        raise TypeError("Deep Agent workflow tool returned an invalid summary")
    messages = response.get("messages", [])
    final_content = messages[-1].content if messages else ""
    summary["final_message"] = (
        final_content if isinstance(final_content, str) else str(final_content)
    )
    return summary
