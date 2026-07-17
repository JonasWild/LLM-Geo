"""Top-level LangGraph orchestration: plan -> implement (parallel) -> assemble/execute -> repair."""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import coder, executor, planner
from .artifacts import RunArtifacts
from .llm import get_model
from .models import DAGSpec, ExecutionResult, NodeImplementation, NodeSpec, RunReport
from .trace import Tracer

MAX_REPAIR_ATTEMPTS = 3


class State(TypedDict):
    task: str
    dag: DAGSpec | None
    implementations: Annotated[dict[str, NodeImplementation], operator.or_]
    implementation_attempts: Annotated[dict[str, int], operator.or_]
    implement_calls: Annotated[int, operator.add]
    attempt: int
    result: ExecutionResult | None
    node_to_implement: NodeSpec | None
    repair_context: dict | None


def build_app(tracer: Tracer, artifacts: RunArtifacts | None = None):
    model = get_model()

    def plan_node(state: State) -> dict:
        with tracer.span("plan"):
            dag = planner.plan(state["task"], model, artifacts=artifacts)
        if artifacts:
            artifacts.save_plan(dag)
        return {
            "dag": dag, "attempt": 0, "implementations": {}, "implementation_attempts": {}, "implement_calls": 0,
        }

    def implement_one(state: State) -> dict:
        node = state["node_to_implement"]
        repair_context = state.get("repair_context")
        with tracer.span("implement", node.id, mode="repair" if repair_context else "initial"):
            impl, attempts = coder.implement_node(node, model, artifacts=artifacts, repair_context=repair_context)
        return {
            "implementations": {node.id: impl},
            "implementation_attempts": {node.id: attempts},
            "implement_calls": 1,
        }

    def assemble_node(state: State) -> dict:
        attempt = state["attempt"] + 1
        # On repair rounds, reuse the previous attempt's outputs for everything that is not the
        # (re-implemented) failing node or downstream of it -- see executor.execute.
        prior = state.get("result")
        prior_outputs = prior.outputs if prior and not prior.success else None
        stale = frozenset(prior.failing_node_ids) if prior_outputs else frozenset()
        with tracer.span("assemble_execute", attempt=attempt):
            result = executor.execute(
                state["dag"], state["implementations"], tracer,
                prior_outputs=prior_outputs, stale_node_ids=stale,
            )
        if artifacts:
            artifacts.save_execution(attempt, result, state["dag"])
        return {"result": result, "attempt": attempt}

    def route_after_plan(state: State):
        pending = [n for n in state["dag"].nodes if not n.registry_id]
        return [Send("implement_one", {"node_to_implement": n}) for n in pending] or "assemble"

    def route_after_assemble(state: State):
        result = state["result"]
        if result.success or state["attempt"] >= MAX_REPAIR_ATTEMPTS:
            return END
        to_repair = [n for n in state["dag"].nodes if n.id in result.failing_node_ids and not n.registry_id]
        return [
            Send("implement_one", {
                "node_to_implement": n,
                "repair_context": {
                    "previous_code": state["implementations"][n.id].code,
                    "error": result.error,
                    "traceback": (result.error_traceback or "")[-2000:],
                },
            })
            for n in to_repair
        ] or END

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("implement_one", implement_one)
    g.add_node("assemble", assemble_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, ["implement_one", "assemble"])
    g.add_edge("implement_one", "assemble")
    g.add_conditional_edges("assemble", route_after_assemble, ["implement_one", END])
    return g.compile()


def run(
    task: str,
    tracer: Tracer | None = None,
    artifacts: RunArtifacts | None = None,
    task_name: str | None = None,
) -> RunReport:
    """Run one task end-to-end. Every run writes a debug bundle (prompts, code attempts, contract
    results, execution errors with full tracebacks) under output/<task_name>/<timestamp>/; pass
    `task_name` for a readable bundle dir name, or a prebuilt `artifacts` to control it fully.
    """
    artifacts = artifacts or RunArtifacts(task_name or task)
    tracer = tracer or Tracer(
        path=artifacts.dir / "trace.jsonl",
        log_file=artifacts.dir / "trace.log",
        on_error=lambda phase, node_id, message, tb: artifacts.record_error(
            phase, message, node_id=node_id, traceback_text=tb
        ),
    )
    artifacts.save_task(task)
    app = build_app(tracer, artifacts)
    agent_graph_mermaid = app.get_graph().draw_mermaid()
    t0 = time.monotonic()
    try:
        with tracer.span("run", task=task[:60]):
            final = app.invoke({"task": task}, {"recursion_limit": 50})
    except Exception:
        artifacts.finalize(None)  # the bundle still holds every prompt/attempt/error so far
        raise
    report = RunReport(
        task=task,
        dag=final["dag"],
        implementations=final["implementations"],
        implementation_attempts=final["implementation_attempts"],
        implement_calls=final["implement_calls"],
        repair_attempts=final["attempt"],
        result=final["result"],
        duration_ms=(time.monotonic() - t0) * 1000,
        agent_graph_mermaid=agent_graph_mermaid,
        artifacts_dir=str(artifacts.dir),
    )
    artifacts.finalize(report)
    return report
