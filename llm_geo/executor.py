"""Assemble validated nodes per the DAG, execute in dependency order, validate the final result."""
from __future__ import annotations

import time
import traceback

import geopandas as gpd
import networkx as nx
import pandas as pd

from .contracts import compile_node
from .models import DAGSpec, ExecutionResult, NodeImplementation, NodeSpec
from .registry import REGISTRY
from .trace import Tracer

# Hard cap on a map node's fan-out: mapping a live-API registry op over an unbounded
# upstream collection is a rate-limit incident waiting to happen, so refuse it up front.
MAX_MAP_ITEMS = 100


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


def _collect_map_outputs(items: list[dict], declared_outputs: dict[str, str]) -> dict:
    """Lift a list of per-element output dicts into one output dict. For each output name, if every
    element produced a GeoDataFrame they are concatenated into one (per-element provenance collected
    into a list under `.attrs["provenance"]`); otherwise the per-element values become a list."""
    if not items:
        # An empty mapped sequence (e.g. an upstream retrieval found nothing) still has to satisfy
        # the node's declared outputs so downstream wiring and the terminal-output check hold.
        return {
            name: (gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326") if "GeoDataFrame" in typ else [])
            for name, typ in declared_outputs.items()
        }
    collected: dict = {}
    for name in items[0]:
        values = [item[name] for item in items if name in item]
        if values and all(isinstance(v, gpd.GeoDataFrame) for v in values):
            merged = pd.concat(values, ignore_index=True)
            provenances = [
                v.attrs["provenance"] for v in values if "provenance" in getattr(v, "attrs", {})
            ]
            if provenances:
                merged.attrs["provenance"] = provenances
            collected[name] = merged
        else:
            collected[name] = values
    return collected


def _execute_map(node: NodeSpec, resolved: dict, tracer: Tracer) -> dict:
    """Run `node`'s trusted registry op once per element of its `map_over` input, broadcasting every
    other resolved input unchanged, and collect the per-element outputs."""
    fn = REGISTRY[node.registry_id]["fn"]
    sequence = resolved.get(node.map_over)
    if not isinstance(sequence, list):
        raise TypeError(
            f"map_over input '{node.map_over}' must resolve to a list, got {type(sequence).__name__}"
        )
    if len(sequence) > MAX_MAP_ITEMS:
        raise ValueError(f"map over {len(sequence)} items exceeds the cap of {MAX_MAP_ITEMS}")
    items: list[dict] = []
    for index, element in enumerate(sequence):
        per_item = {**resolved, node.map_over: element}
        try:
            with tracer.span("exec", node.id, item=index):
                items.append(fn(**per_item))
        except Exception as exc:
            raise RuntimeError(f"map item {index + 1}/{len(sequence)} ({element!r}): {exc}") from exc
    return _collect_map_outputs(items, node.outputs)


def execute(
    dag: DAGSpec,
    implementations: dict[str, NodeImplementation],
    tracer: Tracer,
    prior_outputs: dict[str, dict] | None = None,
    stale_node_ids: frozenset[str] | set[str] = frozenset(),
) -> ExecutionResult:
    """Execute the DAG in dependency order. On repair rounds, pass the previous attempt's
    `outputs` and the re-implemented node ids as `stale_node_ids`: any prior output whose node is
    neither stale nor downstream of a stale node is reused (status `cached`) instead of re-run --
    in particular, live retrieval nodes that already succeeded are not hit again."""
    g = build_graph(dag)
    by_id = {n.id: n for n in dag.nodes}
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible as exc:
        return ExecutionResult(success=False, error=f"DAG has a cycle: {exc}", error_traceback=traceback.format_exc())

    reusable: dict[str, dict] = {}
    if prior_outputs:
        stale = set(stale_node_ids)
        for node_id in stale_node_ids:
            stale |= nx.descendants(g, node_id)
        reusable = {nid: out for nid, out in prior_outputs.items() if nid in by_id and nid not in stale}

    outputs: dict[str, dict] = {}
    node_order: list[str] = []
    node_status: dict[str, str] = {}
    node_duration_ms: dict[str, float] = {}
    node_inputs: dict[str, dict] = {}
    for node_id in order:
        node = by_id[node_id]
        node_order.append(node_id)
        if node_id in reusable:
            with tracer.span("exec", node_id, cached=True):
                outputs[node_id] = reusable[node_id]
            node_status[node_id] = "cached"
            node_duration_ms[node_id] = 0.0
            continue
        t0 = time.monotonic()
        try:
            with tracer.span("exec", node_id):
                node_inputs[node_id] = _resolve_inputs(node, outputs)
                if node.map_over:
                    outputs[node_id] = _execute_map(node, node_inputs[node_id], tracer)
                else:
                    fn = _callable_for(node, implementations)
                    outputs[node_id] = fn(**node_inputs[node_id])
        except Exception as exc:
            node_status[node_id] = "error"
            node_duration_ms[node_id] = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                success=False, outputs=outputs, failing_node_ids=[node_id], error=str(exc),
                error_traceback=traceback.format_exc(),
                node_order=node_order, node_status=node_status, node_duration_ms=node_duration_ms,
                node_inputs=node_inputs,
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
                node_inputs=node_inputs,
            )

    return ExecutionResult(
        success=True, outputs=outputs,
        node_order=node_order, node_status=node_status, node_duration_ms=node_duration_ms,
        node_inputs=node_inputs,
    )
