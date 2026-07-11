"""Workflow-DAG validation, contracts, and visualization."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx

from llm_geo.utils.models import DataSource, WorkflowPlan

if TYPE_CHECKING:
    from llm_geo.operations.registry import RegisteredOperation


def _operation_registry(
    operations: Sequence["RegisteredOperation"],
) -> dict[str, "RegisteredOperation"]:
    """Index operations by canonical short ID and legacy qualified ID."""
    return {
        alias: operation
        for operation in operations
        for alias in (operation.id, operation.qualified_id)
    }


def validate_workflow_plan(
    plan: WorkflowPlan,
    sources: list[DataSource],
    registered_operations: Sequence["RegisteredOperation"] = (),
    *,
    require_result_manifest: bool = False,
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
            if node.implementation == "generated" and graph.in_degree(node.id) == 0:
                issues.append(f"Generated operation {node.id!r} has no input data.")
            if node.implementation == "generated":
                graph_input_ids = set(graph.predecessors(node.id))
                for literal_name in node.literal_arguments:
                    if not literal_name.isidentifier():
                        issues.append(
                            f"Generated operation {node.id!r} has invalid literal "
                            f"parameter name {literal_name!r}."
                        )
                    if literal_name in graph_input_ids:
                        issues.append(
                            f"Generated operation {node.id!r} supplies {literal_name!r} "
                            "both as graph input and literal argument."
                        )
        elif (
            node.implementation != "generated"
            or node.registered_operation_id
            or node.literal_arguments
        ):
            issues.append(f"Data node {node.id!r} cannot select an implementation.")
        if node.kind == "data":
            producer_count = graph.in_degree(node.id)
            if producer_count > 1:
                issues.append(
                    f"Data node {node.id!r} has {producer_count} producing operations; "
                    "each data node must have at most one producer."
                )
            if producer_count == 0 and not node.data_path:
                issues.append(
                    f"Data node {node.id!r} has no producing operation and no "
                    "existing source path."
                )
    sinks = [node for node in plan.nodes if graph.out_degree(node.id) == 0]
    if not sinks or any(node.kind != "data" for node in sinks):
        issues.append("Every workflow sink must be a data/result node.")
    if require_result_manifest:
        manifests = [
            node
            for node in plan.nodes
            if node.kind == "data"
            and Path(node.data_path).name == "llm_geo_result.json"
        ]
        if not manifests:
            issues.append(
                "The executable workflow must include an operation that writes "
                "llm_geo_result.json and an output data node whose data_path names "
                "that manifest."
            )
        elif any(graph.in_degree(node.id) != 1 for node in manifests):
            issues.append(
                "Every llm_geo_result.json data node must be produced by exactly "
                "one explicit operation."
            )
    planned_paths = {node.data_path for node in plan.nodes if node.data_path}
    for source in sources:
        if source.location not in planned_paths:
            issues.append(f"Provided data source is absent from the plan: {source.location}")
    registry = _operation_registry(registered_operations)
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
        graph_input_ids = list(graph.predecessors(node.id))
        if operation.category == "retrieval" and graph_input_ids:
            issues.append(
                f"Retrieval operation {node.id!r} must be a root operation with no "
                f"graph inputs, but receives {graph_input_ids}. Supply its request "
                "and configuration through literal_arguments."
            )
        if operation.category == "retrieval" and "output_path" in node.literal_arguments:
            output_paths = [
                node_map[output_id].data_path
                for output_id in graph.successors(node.id)
                if node_map[output_id].data_path
            ]
            expected_output_path = node.literal_arguments["output_path"]
            if output_paths and output_paths != [expected_output_path]:
                issues.append(
                    f"Retrieval operation {node.id!r} writes {expected_output_path!r}, "
                    f"but its output data node declares {output_paths}. Add any "
                    "required conversion or rendering operation before a different "
                    "output format."
                )
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
                f"Registered operation {node.id!r} has {graph_input_count} graph "
                f"inputs {graph_input_ids}, but {operation.id!r} has only "
                f"{len(available_graph_parameters)} parameters available for graph "
                f"binding: {available_graph_parameters}. Parameters supplied as "
                f"literals: {sorted(literal_names)}."
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
    node = node_map[operation_id]
    # Registered-operation bridges bind their literals internally. Generated
    # functions expose task-known literals as explicit parameters instead.
    literal_arguments = (
        dict(node.literal_arguments) if node.implementation == "generated" else {}
    )
    parameters = [*inputs, *literal_arguments]
    return {
        "node_id": operation_id,
        "description": node_map[operation_id].description,
        "inputs": inputs,
        "literal_arguments": literal_arguments,
        "outputs": outputs,
        "signature": f"def {operation_id}({', '.join(parameters)}):",
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


def validate_operation_code(
    code: str,
    contract: dict[str, Any],
    *,
    generated: bool = True,
) -> list[str]:
    """Validate one operation as an isolated, import-safe Python function."""
    try:
        module = ast.parse(code)
    except SyntaxError as error:
        return [f"Operation code is not valid Python: {error.msg} at line {error.lineno}."]

    operation_id = str(contract["node_id"])
    functions = [
        statement
        for statement in module.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == operation_id
    ]
    issues: list[str] = []
    if len(functions) != 1:
        issues.append(
            f"Code must define operation function {operation_id!r} exactly once."
        )
        return issues
    function = functions[0]
    if isinstance(function, ast.AsyncFunctionDef):
        issues.append("Operation functions must be synchronous.")

    expected_parameters = [
        *contract.get("inputs", []),
        *contract.get("literal_arguments", {}),
    ]
    actual_parameters = [
        argument.arg for argument in [*function.args.posonlyargs, *function.args.args]
    ]
    if (
        actual_parameters != expected_parameters
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
    ):
        issues.append(
            f"Function parameters must be exactly {expected_parameters}, found "
            f"{actual_parameters}."
        )

    if generated:
        unexpected_top_level = [
            type(statement).__name__
            for statement in module.body
            if statement is not function
            and not isinstance(statement, (ast.Import, ast.ImportFrom))
        ]
        if unexpected_top_level:
            issues.append(
                "Generated operation modules may contain only imports and the "
                f"operation function; found {unexpected_top_level}."
            )
        returns = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
        ]
        if not returns:
            issues.append("Generated operation must contain a return statement.")
        expected_outputs = list(contract.get("outputs", []))
        final_statement = function.body[-1] if function.body else None
        returned_names: list[str] = []
        if isinstance(final_statement, ast.Return):
            if isinstance(final_statement.value, ast.Name):
                returned_names = [final_statement.value.id]
            elif isinstance(final_statement.value, (ast.Tuple, ast.List)):
                returned_names = [
                    element.id
                    for element in final_statement.value.elts
                    if isinstance(element, ast.Name)
                ]
        if returned_names != expected_outputs:
            issues.append(
                "Function must end by returning output variables exactly in graph "
                f"order {expected_outputs}, found {returned_names}."
            )
    return issues


def render_main(plan: WorkflowPlan) -> str:
    """Render explicit deterministic execution of every operation in the plan."""
    graph = plan_to_graph(plan)
    node_map = {node.id: node for node in plan.nodes}
    source_data = [
        node
        for node in plan.nodes
        if node.kind == "data" and graph.in_degree(node.id) == 0
    ]
    lines: list[str] = []
    if source_data:
        lines.extend(["import geopandas as gpd", ""])
    lines.append("def main():")
    for node in source_data:
        lines.append(f"    # existing source -> {node.id}")
        lines.append(f"    {node.id} = gpd.read_file({node.data_path!r})")
        lines.append("")

    for node_id in nx.topological_sort(graph):
        if node_map[node_id].kind != "operation":
            continue
        contract = operation_contract(plan, node_id)
        inputs = list(contract["inputs"])
        outputs = list(contract["outputs"])
        literals = dict(contract["literal_arguments"])
        display_inputs = ", ".join(inputs) or "()"
        display_outputs = ", ".join(outputs)
        lines.append(
            f"    # {display_inputs} -> {node_id} -> {display_outputs}"
        )
        arguments = [*inputs, *(f"{key}={value!r}" for key, value in literals.items())]
        call = f"{node_id}({', '.join(arguments)})"
        assignment = ", ".join(outputs)
        lines.append(f"    {assignment} = {call}")
        lines.append("")

    sinks = [
        node.id
        for node in plan.nodes
        if node.kind == "data" and graph.out_degree(node.id) == 0
    ]
    if len(sinks) == 1:
        lines.append(f"    return {sinks[0]}")
    else:
        lines.append(f"    return {', '.join(sinks)}")
    lines.extend(["", "", "main()", ""])
    return "\n".join(lines)


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
        f"{operation.import_statement}\n\n"
        f"def {operation_id}({arguments}):\n"
        f"    return {operation.name}({', '.join(call_arguments)})"
    )
