"""Top-level LangGraph orchestration: plan -> implement (parallel) -> assemble/execute -> repair."""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import coder, executor, planner
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


def build_app(tracer: Tracer):
    model = get_model()

    def plan_node(state: State) -> dict:
        with tracer.span("plan"):
            dag = planner.plan(state["task"], model)
        return {
            "dag": dag, "attempt": 0, "implementations": {}, "implementation_attempts": {}, "implement_calls": 0,
        }

    def implement_one(state: State) -> dict:
        node = state["node_to_implement"]
        with tracer.span("implement", node.id):
            impl, attempts = coder.implement_node(node, model)
        return {
            "implementations": {node.id: impl},
            "implementation_attempts": {node.id: attempts},
            "implement_calls": 1,
        }

    def assemble_node(state: State) -> dict:
        with tracer.span("assemble_execute", attempt=state["attempt"] + 1):
            result = executor.execute(state["dag"], state["implementations"], tracer)
        return {"result": result, "attempt": state["attempt"] + 1}

    def route_after_plan(state: State):
        pending = [n for n in state["dag"].nodes if not n.registry_id]
        return [Send("implement_one", {"node_to_implement": n}) for n in pending] or "assemble"

    def route_after_assemble(state: State):
        result = state["result"]
        if result.success or state["attempt"] >= MAX_REPAIR_ATTEMPTS:
            return END
        to_repair = [n for n in state["dag"].nodes if n.id in result.failing_node_ids and not n.registry_id]
        return [Send("implement_one", {"node_to_implement": n}) for n in to_repair] or END

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("implement_one", implement_one)
    g.add_node("assemble", assemble_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, ["implement_one", "assemble"])
    g.add_edge("implement_one", "assemble")
    g.add_conditional_edges("assemble", route_after_assemble, ["implement_one", END])
    return g.compile()


def run(task: str, tracer: Tracer | None = None) -> RunReport:
    tracer = tracer or Tracer()
    app = build_app(tracer)
    agent_graph_mermaid = app.get_graph().draw_mermaid()
    t0 = time.monotonic()
    with tracer.span("run", task=task[:60]):
        final = app.invoke({"task": task}, {"recursion_limit": 50})
    return RunReport(
        task=task,
        dag=final["dag"],
        implementations=final["implementations"],
        implementation_attempts=final["implementation_attempts"],
        implement_calls=final["implement_calls"],
        repair_attempts=final["attempt"],
        result=final["result"],
        duration_ms=(time.monotonic() - t0) * 1000,
        agent_graph_mermaid=agent_graph_mermaid,
    )
