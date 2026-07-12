"""Assemble validated nodes per the DAG, execute in dependency order, validate the final result."""
from __future__ import annotations

import time

import networkx as nx

from .contracts import compile_node
from .models import DAGSpec, ExecutionResult, NodeImplementation, NodeSpec
from .registry import REGISTRY
from .trace import Tracer


def build_graph(dag: DAGSpec) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(n.id for n in dag.nodes)
    g.add_edges_from((dep, n.id) for n in dag.nodes for dep in n.depends_on)
    return g


def _resolve_inputs(node: NodeSpec, outputs: dict[str, dict]) -> dict:
    """An input is fed by whichever dependency produced an output of the same name. If several
    inputs/dependencies are left unmatched by name (e.g. two dependencies both output `features`),
    pair the remaining ones positionally in `depends_on` order.
    """
    resolved = dict(node.params)
    consumed, unresolved = set(), []
    for name in node.inputs:
        dep = next((d for d in node.depends_on if d not in consumed and name in outputs.get(d, {})), None)
        if dep is not None:
            resolved[name] = outputs[dep][name]
            consumed.add(dep)
        elif name not in resolved:
            unresolved.append(name)

    remaining = [d for d in node.depends_on if d not in consumed]
    for name, dep in zip(unresolved, remaining):
        dep_out = outputs.get(dep, {})
        if len(dep_out) == 1:
            resolved[name] = next(iter(dep_out.values()))
    return resolved


def _callable_for(node: NodeSpec, implementations: dict[str, NodeImplementation]):
    if node.registry_id:
        return REGISTRY[node.registry_id]["fn"]
    return compile_node(implementations[node.id].code)


def execute(dag: DAGSpec, implementations: dict[str, NodeImplementation], tracer: Tracer) -> ExecutionResult:
    g = build_graph(dag)
    by_id = {n.id: n for n in dag.nodes}
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible as exc:
        return ExecutionResult(success=False, error=f"DAG has a cycle: {exc}")

    outputs: dict[str, dict] = {}
    node_order: list[str] = []
    node_status: dict[str, str] = {}
    node_duration_ms: dict[str, float] = {}
    for node_id in order:
        node = by_id[node_id]
        node_order.append(node_id)
        t0 = time.monotonic()
        try:
            with tracer.span("exec", node_id):
                fn = _callable_for(node, implementations)
                outputs[node_id] = fn(**_resolve_inputs(node, outputs))
        except Exception as exc:
            node_status[node_id] = "error"
            node_duration_ms[node_id] = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                success=False, outputs=outputs, failing_node_ids=[node_id], error=str(exc),
                node_order=node_order, node_status=node_status, node_duration_ms=node_duration_ms,
            )
        node_status[node_id] = "ok"
        node_duration_ms[node_id] = (time.monotonic() - t0) * 1000

    for node_id in (n for n in order if g.out_degree(n) == 0):
        missing = [name for name in by_id[node_id].outputs if name not in outputs.get(node_id, {})]
        if missing:
            node_status[node_id] = "error"
            return ExecutionResult(
                success=False, outputs=outputs, failing_node_ids=[node_id],
                error=f"terminal node '{node_id}' missing outputs {missing}",
                node_order=node_order, node_status=node_status, node_duration_ms=node_duration_ms,
            )

    return ExecutionResult(
        success=True, outputs=outputs,
        node_order=node_order, node_status=node_status, node_duration_ms=node_duration_ms,
    )
