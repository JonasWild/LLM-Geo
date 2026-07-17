"""Single-step planning: task -> validated DAGSpec, via a deep agent with structured output.

The returned plan is statically validated (`validate_dag`) before any node is implemented, so
wiring/type errors are corrected by the planner instead of burning coder + execution rounds.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .llm import retry_on_rate_limit
from .models import DAGSpec, NodeSpec
from .registry import REGISTRY, catalog_text
from .structured_output import run_structured_agent

MAX_PLAN_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a geospatial analysis planner. Given a user task, produce the COMPLETE \
workflow as a single DAG in one shot -- do not ask questions, do not explore, just plan.

Rules:
- Classify every node's `kind` as exactly one of: retrieval, transformation, synthesis.
- Every node needs a unique snake_case id, a precise `description`, `depends_on` node ids, \
`inputs`/`outputs` port specs, and literal `params` for anything not coming from a dependency.
- Each `inputs`/`outputs` entry maps a name to a port spec: `type` (one of str, int, float, bool, \
dict, GeoDataFrame) plus a meaningful `description` (what the value means, units, expectations). \
For GeoDataFrame ports also declare, whenever known, the required `columns` (name -> dtype), the \
`geometry` type (Point|LineString|Polygon|any) and `crs` -- these are validated at runtime on both \
the producer and the consumer, so declare exactly what is required, no more. For scalar/dict \
ports include a realistic `example`.
- Retrieval nodes must output a `GeoDataFrame`-typed value carrying provenance metadata in \
`.attrs["provenance"]`.
- Wiring: a node's input is fed by whichever of its `depends_on` produced an output of the \
SAME name. Name inputs/outputs consistently across dependent nodes so this wiring resolves, and \
give both ends of an edge the SAME type and constraints.
- A trusted implementation registry is available. If a node's need is exactly covered by one \
of these, set `registry_id` to that id and declare the node's inputs/outputs with the same names \
and types (inputs with a default may be omitted; other inputs must be declared or covered by \
`params`). Otherwise leave `registry_id` null and a coding agent will implement it from your \
description and port specs.

Trusted registry:
""" + catalog_text() + "\n\nReturn only the final DAGSpec."


def _validate_registry_node(node: NodeSpec, errors: list[str]) -> None:
    spec = REGISTRY.get(node.registry_id)
    if spec is None:
        errors.append(f"node '{node.id}': unknown registry_id '{node.registry_id}'")
        return
    declared_outputs = {name: port.type for name, port in node.outputs.items()}
    if declared_outputs != spec["outputs"]:
        errors.append(
            f"node '{node.id}': outputs {declared_outputs} must exactly match "
            f"registry '{node.registry_id}' outputs {spec['outputs']}"
        )
    for name, port in node.inputs.items():
        if name not in spec["inputs"]:
            errors.append(f"node '{node.id}': registry '{node.registry_id}' has no input '{name}'")
        elif port.type != spec["inputs"][name]:
            errors.append(
                f"node '{node.id}': input '{name}' is {port.type} but registry "
                f"'{node.registry_id}' expects {spec['inputs'][name]}"
            )
    for name in spec["inputs"]:
        if name not in spec["defaults"] and name not in node.inputs and name not in node.params:
            errors.append(
                f"node '{node.id}': registry '{node.registry_id}' requires input '{name}' "
                "(declare it as an input fed by a dependency, or provide it in params)"
            )


def validate_dag(dag: DAGSpec) -> list[str]:
    """Static plan checks mirroring the executor's wiring: every input must have a source of the
    same declared type, and registry-backed nodes must match the registry contract."""
    errors: list[str] = []
    by_id: dict[str, NodeSpec] = {}
    for node in dag.nodes:
        if node.id in by_id:
            errors.append(f"duplicate node id '{node.id}'")
        by_id[node.id] = node

    for node in dag.nodes:
        for dep in node.depends_on:
            if dep not in by_id:
                errors.append(f"node '{node.id}' depends on unknown node '{dep}'")
        if node.registry_id:
            _validate_registry_node(node, errors)

        deps = [d for d in node.depends_on if d in by_id]
        consumed: set[str] = set()
        unmatched: list[str] = []
        for name, port in node.inputs.items():
            dep = next((d for d in deps if d not in consumed and name in by_id[d].outputs), None)
            if dep is not None:
                consumed.add(dep)
                upstream = by_id[dep].outputs[name]
                if upstream.type != port.type:
                    errors.append(
                        f"edge '{dep}' -> '{node.id}': '{name}' is produced as {upstream.type} "
                        f"but consumed as {port.type}"
                    )
            elif name not in node.params:
                unmatched.append(name)
        remaining = [d for d in deps if d not in consumed]
        for name, dep in zip(unmatched, remaining):
            outputs = by_id[dep].outputs
            if len(outputs) != 1:
                errors.append(
                    f"node '{node.id}': input '{name}' has no same-named upstream output and "
                    f"cannot be positionally resolved from '{dep}'"
                )
                continue
            upstream = next(iter(outputs.values()))
            if upstream.type != node.inputs[name].type:
                errors.append(
                    f"edge '{dep}' -> '{node.id}': '{name}' positionally resolves to a "
                    f"{upstream.type} output but is declared {node.inputs[name].type}"
                )
        for name in unmatched[len(remaining):]:
            errors.append(f"node '{node.id}': input '{name}' has no source (params or dependency output)")
    return errors


@retry_on_rate_limit
def plan(task: str, model: BaseChatModel) -> DAGSpec:
    content = task
    errors: list[str] = []
    for _ in range(MAX_PLAN_ATTEMPTS):
        dag, _ = run_structured_agent(model, SYSTEM_PROMPT, content, DAGSpec)
        errors = validate_dag(dag)
        if not errors:
            return dag
        content = (
            task + "\n\nYour previous DAGSpec failed static validation:\n- " + "\n- ".join(errors)
            + "\nReturn a corrected DAGSpec."
        )
    raise ValueError(
        f"planner produced an invalid DAG after {MAX_PLAN_ATTEMPTS} attempts:\n- " + "\n- ".join(errors)
    )
