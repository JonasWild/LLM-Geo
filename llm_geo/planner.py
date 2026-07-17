"""Single-step planning: task -> validated DAGSpec, via a deep agent with structured output."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel

from .artifacts import RunArtifacts
from .contracts import coarse_type_ok
from .executor import resolve_inputs
from .llm import retry_on_rate_limit
from .models import DAGSpec
from .registry import REGISTRY, catalog_text
from .structured_output import run_structured_agent

SYSTEM_PROMPT = ("""You are a geospatial analysis planner. Given a user task, produce the COMPLETE \
workflow as a single DAG in one shot -- do not ask questions, do not explore, just plan.

Rules:
- Classify every node's `kind` as exactly one of: retrieval, transformation, synthesis.
- Every node needs a unique snake_case id, a precise `description`, `depends_on` node ids, \
`inputs`/`outputs` (name -> port spec), and literal `params` for anything not coming from a \
dependency.
- A port spec has three fields:
  * `type`: one of str, int, float, bool, dict, GeoDataFrame.
  * `description`: precisely what the value means -- units, semantics, expected dict keys, \
expected GeoDataFrame columns and geometry kind. The implementer sees ONLY this, so be specific.
  * `example` (optional): a realistic literal sample value for scalar/dict ports, never for \
GeoDataFrame. Provide one whenever you can -- it becomes the value the implementation is tested \
against.
- The nodes must form exactly ONE connected workflow graph: every node must be linked to the \
rest via `depends_on` (no disconnected islands, no parallel unrelated graphs), converging on \
the final answer node(s) for the task.
- Retrieval nodes must output a `GeoDataFrame`-typed value carrying provenance metadata in \
`.attrs["provenance"]`.
- Wiring: a node's input is fed by whichever of its `depends_on` produced an output of the \
SAME name. Name inputs/outputs consistently across dependent nodes so this wiring resolves, \
and give the connected output and input the same `type`.
- A trusted implementation registry is available. If a node's need is exactly covered by one \
of these, set `registry_id` to that id and make the node's inputs/outputs match it exactly \
(same names, same types). Otherwise leave `registry_id` null and a coding agent will implement \
it from your description.

Trusted registry:
""" + catalog_text()
+ """\n\nYou will always try to first find a possible node from the trusted registry!
\n\nReturn only the final DAGSpec.
""")


def single_graph_errors(dag: DAGSpec) -> list[str]:
    """Checks that the plan is exactly one graph: no duplicate/unknown ids, no disconnected
    islands. Returns human-readable violations (empty = valid)."""
    ids = [n.id for n in dag.nodes]
    if not ids:
        return ["the DAG has no nodes"]
    errors = [f"duplicate node id '{i}'" for i in sorted({i for i in ids if ids.count(i) > 1})]
    known = set(ids)
    for node in dag.nodes:
        for dep in node.depends_on:
            if dep not in known:
                errors.append(f"node '{node.id}' depends on unknown node '{dep}'")
    if errors or len(ids) == 1:
        return errors
    g = nx.Graph()
    g.add_nodes_from(known)
    g.add_edges_from((dep, n.id) for n in dag.nodes for dep in n.depends_on)
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    if len(components) > 1:
        islands = "; ".join("{" + ", ".join(sorted(c)) + "}" for c in components)
        errors.append(f"the DAG splits into {len(components)} disconnected graphs: {islands}")
    return errors


@dataclass(frozen=True)
class _TypeToken:
    """Stand-in for a dependency's output value when dry-running input resolution on types."""

    type: str


def _assignable(src: str, dst: str) -> bool:
    return src == dst or (src == "int" and dst == "float")


def wiring_errors(dag: DAGSpec) -> list[str]:
    """Dry-run the executor's exact input resolution with type tokens instead of values: every
    input must be fed, and fed with a compatible coarse type. Catches at plan time the mismatches
    that would otherwise surface as confusing crashes inside downstream nodes."""
    known = {n.id for n in dag.nodes}
    type_outputs = {
        n.id: {name: _TypeToken(port.type) for name, port in n.outputs.items()} for n in dag.nodes
    }
    errors = []
    for node in dag.nodes:
        if any(dep not in known for dep in node.depends_on):
            continue  # unknown deps are already reported by single_graph_errors
        resolved = resolve_inputs(node, {dep: type_outputs[dep] for dep in node.depends_on})
        for name, port in node.inputs.items():
            if name not in resolved:
                errors.append(
                    f"node '{node.id}': input '{name}' is fed by no dependency output and no param"
                )
            elif isinstance(fed := resolved[name], _TypeToken):
                if not _assignable(fed.type, port.type):
                    producer = next(
                        (dep for dep in node.depends_on if name in type_outputs[dep]), "a dependency"
                    )
                    errors.append(
                        f"node '{node.id}': input '{name}' expects {port.type} but '{producer}' "
                        f"outputs {fed.type}"
                    )
            elif not coarse_type_ok(fed, port.type):
                errors.append(
                    f"node '{node.id}': input '{name}' expects {port.type} but its param literal "
                    f"is {type(fed).__name__} ({fed!r})"
                )
    return errors


def registry_errors(dag: DAGSpec) -> list[str]:
    """Nodes bound to a trusted registry operation must mirror its contract exactly."""
    errors = []
    for node in dag.nodes:
        if not node.registry_id:
            continue
        spec = REGISTRY.get(node.registry_id)
        if spec is None:
            errors.append(f"node '{node.id}': unknown registry_id '{node.registry_id}'")
            continue
        for name, port in node.inputs.items():
            if name not in spec["inputs"]:
                errors.append(
                    f"node '{node.id}': registry op '{node.registry_id}' has no input '{name}' "
                    f"(available: {sorted(spec['inputs'])})"
                )
            elif spec["inputs"][name] != port.type:
                errors.append(
                    f"node '{node.id}': registry input '{name}' has type {spec['inputs'][name]}, "
                    f"not {port.type}"
                )
        for name in set(spec["outputs"]) - set(node.outputs):
            errors.append(
                f"node '{node.id}': registry op '{node.registry_id}' outputs '{name}', declare it"
            )
        for name, port in node.outputs.items():
            if name not in spec["outputs"]:
                errors.append(
                    f"node '{node.id}': registry op '{node.registry_id}' has no output '{name}' "
                    f"(it outputs: {sorted(spec['outputs'])})"
                )
            elif spec["outputs"][name] != port.type:
                errors.append(
                    f"node '{node.id}': registry output '{name}' has type {spec['outputs'][name]}, "
                    f"not {port.type}"
                )
    return errors


def plan_errors(dag: DAGSpec) -> list[str]:
    errors = single_graph_errors(dag)
    return errors if errors else wiring_errors(dag) + registry_errors(dag)


@retry_on_rate_limit
def plan(task: str, model: BaseChatModel, artifacts: RunArtifacts | None = None) -> DAGSpec:
    if artifacts:
        artifacts.save_planner_prompts(SYSTEM_PROMPT, task)
    dag, _ = run_structured_agent(model, SYSTEM_PROMPT, task, DAGSpec)
    errors = plan_errors(dag)
    if not errors:
        return dag

    # One corrective round: show the planner its own invalid plan plus the specific violations.
    retry_prompt = (
        f"{task}\n\nYour previous plan was invalid:\n- " + "\n- ".join(errors)
        + f"\n\nPrevious plan:\n{dag.model_dump_json()}\n\n"
        "Return a corrected DAGSpec that fixes every violation and whose nodes form exactly ONE "
        "connected workflow graph."
    )
    if artifacts:
        artifacts.record_error("plan_validation", "planner produced an invalid DAG, retrying once:\n- " + "\n- ".join(errors))
        artifacts.write("plan/prompts/user_retry.md", retry_prompt)
    dag, _ = run_structured_agent(model, SYSTEM_PROMPT, retry_prompt, DAGSpec)
    remaining = plan_errors(dag)
    if remaining and artifacts:
        artifacts.record_error(
            "plan_validation", "planner DAG is still invalid after one retry, proceeding anyway:\n- " + "\n- ".join(remaining)
        )
    return dag
