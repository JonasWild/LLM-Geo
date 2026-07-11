"""Workflow-DAG validation, contracts, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx

from llm_geo.utils.models import DataSource, WorkflowPlan

if TYPE_CHECKING:
    from llm_geo.operations.registry import RegisteredOperation


def validate_workflow_plan(
    plan: WorkflowPlan,
    sources: list[DataSource],
    registered_operations: Sequence["RegisteredOperation"] = (),
) -> list[str]:
    issues: list[str] = []
    node_ids = [node.id for node in plan.nodes]
    if len(node_ids) != len(set(node_ids)):
        issues.append("Node IDs must be unique.")
    node_map = {node.id: node for node in plan.nodes}
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    for edge in plan.edges:
        if edge.source not in node_map or edge.target not in node_map:
            issues.append(
                f"Edge {edge.source!r}->{edge.target!r} references a missing node."
            )
            continue
        if edge.source == edge.target:
            issues.append(f"Self edge on {edge.source!r} is forbidden.")
            continue
        graph.add_edge(edge.source, edge.target)
        if node_map[edge.source].kind == node_map[edge.target].kind:
            issues.append(
                f"Edge {edge.source!r}->{edge.target!r} must alternate data and operation nodes."
            )
    if graph.number_of_nodes() and not nx.is_directed_acyclic_graph(graph):
        issues.append("The workflow must be a directed acyclic graph.")
    if graph.number_of_nodes() and not nx.is_weakly_connected(graph):
        issues.append("The workflow must be weakly connected.")
    for node in plan.nodes:
        if node.kind == "operation":
            if graph.out_degree(node.id) == 0:
                issues.append(f"Operation {node.id!r} has no output data.")
            if node.implementation == "registered" and not node.registered_operation_id:
                issues.append(f"Registered operation {node.id!r} has no operation ID.")
            if node.implementation == "generated" and node.registered_operation_id:
                issues.append(f"Generated operation {node.id!r} has a registered operation ID.")
            if node.implementation == "generated" and node.literal_arguments:
                issues.append(
                    f"Generated operation {node.id!r} cannot use literal arguments."
                )
            if node.implementation == "generated" and graph.in_degree(node.id) == 0:
                issues.append(f"Generated operation {node.id!r} has no input data.")
        elif (
            node.implementation != "generated"
            or node.registered_operation_id
            or node.literal_arguments
        ):
            issues.append(f"Data node {node.id!r} cannot select an implementation.")
    sinks = [node for node in plan.nodes if graph.out_degree(node.id) == 0]
    if not sinks or any(node.kind != "data" for node in sinks):
        issues.append("Every workflow sink must be a data/result node.")
    planned_paths = {node.data_path for node in plan.nodes if node.data_path}
    for source in sources:
        if source.location not in planned_paths:
            issues.append(f"Provided data source is absent from the plan: {source.location}")
    registry = {operation.id: operation for operation in registered_operations}
    for node in plan.nodes:
        if node.implementation != "registered" or not node.registered_operation_id:
            continue
        operation = registry.get(node.registered_operation_id)
        if operation is None:
            issues.append(
                f"Operation {node.id!r} selects unknown registered operation "
                f"{node.registered_operation_id!r}."
            )
            continue
        parameter_names = [name for name, _, _ in operation.inputs]
        literal_names = set(node.literal_arguments)
        unknown_literals = literal_names - set(parameter_names)
        if unknown_literals:
            issues.append(
                f"Registered operation {node.id!r} has unknown literal arguments: "
                + ", ".join(sorted(unknown_literals))
            )
        available_graph_parameters = [
            name for name in parameter_names if name not in literal_names
        ]
        graph_input_count = graph.in_degree(node.id)
        if graph_input_count > len(available_graph_parameters):
            issues.append(
                f"Registered operation {node.id!r} has too many graph inputs."
            )
        else:
            graph_bound = set(available_graph_parameters[:graph_input_count])
            missing_required = [
                name
                for name in parameter_names
                if name not in literal_names
                and name not in graph_bound
                and name not in operation.defaults
            ]
            if missing_required:
                issues.append(
                    f"Registered operation {node.id!r} is missing required arguments: "
                    + ", ".join(missing_required)
                )
        if graph.out_degree(node.id) != 1:
            issues.append(
                f"Registered operation {node.id!r} returns one output but its graph "
                f"node has {graph.out_degree(node.id)}."
            )
    return issues


def plan_to_graph(plan: WorkflowPlan) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in plan.nodes:
        graph.add_node(
            node.id,
            node_type=node.kind,
            description=node.description,
            data_path=node.data_path,
        )
    graph.add_edges_from((edge.source, edge.target) for edge in plan.edges)
    return graph


def write_graph_artifacts(
    plan: WorkflowPlan, save_dir: Path, task_name: str
) -> list[str]:
    """Write GraphML plus static and interactive workflow visualizations."""
    graph = plan_to_graph(plan)
    workflow_directory = save_dir / "workflow"
    workflow_directory.mkdir(parents=True, exist_ok=True)
    plan_path = workflow_directory / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    graphml_path = workflow_directory / "graph.graphml"
    nx.write_graphml(graph, graphml_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [
        "deepskyblue" if graph.nodes[node]["node_type"] == "operation" else "lightgreen"
        for node in graph.nodes
    ]
    figure, axis = plt.subplots(figsize=(14, 9))
    nx.draw_networkx(
        graph,
        pos=nx.spring_layout(graph, seed=42),
        ax=axis,
        node_color=colors,
        node_size=2200,
        font_size=8,
        arrows=True,
        arrowsize=18,
    )
    axis.set_title("LLM-GEO solution graph")
    axis.set_axis_off()
    png_path = workflow_directory / "graph.png"
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    from pyvis.network import Network

    network = Network(directed=True, height="800px", cdn_resources="remote")
    network.from_nx(graph)
    source_nodes = {node for node in graph if graph.in_degree(node) == 0}
    sink_nodes = {node for node in graph if graph.out_degree(node) == 0}
    for node in network.nodes:
        node_id = node["id"]
        if graph.nodes[node_id]["node_type"] == "operation":
            color = "deepskyblue"
        elif node_id in sink_nodes:
            color = "violet"
        elif node_id in source_nodes:
            color = "lightgreen"
        else:
            color = "orange"
        node.update({"color": color, "shape": "dot", "font": {"size": 20}})
    html_path = workflow_directory / "graph.html"
    network.write_html(str(html_path), notebook=False, open_browser=False)
    return [str(plan_path), str(graphml_path), str(png_path), str(html_path)]


def operation_contract(plan: WorkflowPlan, operation_id: str) -> dict[str, Any]:
    graph = plan_to_graph(plan)
    node_map = {node.id: node for node in plan.nodes}
    inputs = list(graph.predecessors(operation_id))
    outputs = list(graph.successors(operation_id))
    return {
        "node_id": operation_id,
        "description": node_map[operation_id].description,
        "inputs": inputs,
        "outputs": outputs,
        "signature": f"def {operation_id}({', '.join(inputs)}):",
        "return_statement": f"return {', '.join(outputs)}",
    }


def operation_context(plan: WorkflowPlan, operation_id: str) -> dict[str, Any]:
    """Return compact local interfaces needed to implement one operation."""
    graph = plan_to_graph(plan)
    node_map = {node.id: node for node in plan.nodes}
    input_ids = list(graph.predecessors(operation_id))
    output_ids = list(graph.successors(operation_id))
    predecessor_ids = sorted(
        {
            predecessor
            for data_id in input_ids
            for predecessor in graph.predecessors(data_id)
            if node_map[predecessor].kind == "operation"
        }
    )
    successor_ids = sorted(
        {
            successor
            for data_id in output_ids
            for successor in graph.successors(data_id)
            if node_map[successor].kind == "operation"
        }
    )
    return {
        "contract": operation_contract(plan, operation_id),
        "input_data": [
            node_map[node_id].model_dump(mode="json") for node_id in input_ids
        ],
        "output_data": [
            node_map[node_id].model_dump(mode="json") for node_id in output_ids
        ],
        "predecessor_contracts": [
            operation_contract(plan, node_id) for node_id in predecessor_ids
        ],
        "successor_contracts": [
            operation_contract(plan, node_id) for node_id in successor_ids
        ],
    }


def registered_operation_bridge(
    plan: WorkflowPlan,
    operation_id: str,
    operation: "RegisteredOperation",
) -> str:
    """Render a thin wrapper binding graph inputs and planner-supplied literals."""
    contract = operation_contract(plan, operation_id)
    node = next(node for node in plan.nodes if node.id == operation_id)
    graph_arguments = iter(contract["inputs"])
    call_arguments: list[str] = []
    for parameter_name, _, _ in operation.inputs:
        if parameter_name in node.literal_arguments:
            call_arguments.append(
                f"{parameter_name}={node.literal_arguments[parameter_name]!r}"
            )
            continue
        try:
            graph_argument = next(graph_arguments)
        except StopIteration:
            continue
        call_arguments.append(f"{parameter_name}={graph_argument}")
    arguments = ", ".join(contract["inputs"])
    return (
        f"from {operation.module} import {operation.name}\n\n"
        f"def {operation_id}({arguments}):\n"
        f"    return {operation.name}({', '.join(call_arguments)})"
    )
