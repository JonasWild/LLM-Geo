"""Single-step planning: task -> validated DAGSpec, via a deep agent with structured output."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel

from .artifacts import RunArtifacts
from .contracts import coarse_type_ok, dict_field_errors
from .executor import resolve_inputs
from .llm import retry_on_rate_limit
from .models import DAGSpec, PortSpec
from .registry import REGISTRY, catalog_text
from .structured_output import run_structured_agent

SYSTEM_PROMPT = ("""You are a geospatial analysis planner. Given a user task, produce the COMPLETE \
workflow as a single DAG in one shot -- do not ask questions, do not explore, just plan.

Rules:
- Classify every node's `kind` as exactly one of: retrieval, transformation, synthesis.
- Every node needs a unique snake_case id, a precise `description`, `depends_on` node ids, \
`inputs`/`outputs` (name -> port spec), and literal `params` for anything not coming from a \
dependency.
- A port spec has these fields:
  * `type`: one of str, int, float, bool, dict, GeoDataFrame.
  * `description`: precisely what the value means -- units, semantics, expected GeoDataFrame \
columns and geometry kind. The implementer sees ONLY this, so be specific.
  * `fields` (REQUIRED for every dict port): the dict's exact contract, one entry per key: \
key name -> {type, description}. The key type is one of str, int, float, bool, dict, list[str], \
list[int], list[float], list[dict]. Enumerate every key the consumer needs -- implementations \
are validated key by key against this contract, and undeclared keys are rejected. `fields` \
describe the keys INSIDE the port's value: never repeat the port's own name as its only field \
(a port `coordinates` has fields like lat/lon, not a field `coordinates`).
  * `example` (optional): a realistic literal sample value for scalar/dict ports, never for \
GeoDataFrame. Provide one whenever you can -- it becomes the value the implementation is tested \
against. A dict example must match the declared `fields`.
- The nodes must form exactly ONE connected workflow graph: every node must be linked to the \
rest via `depends_on` (no disconnected islands, no parallel unrelated graphs), converging on \
the final answer node(s) for the task.
- Retrieval nodes must output a `GeoDataFrame`-typed value carrying provenance metadata in \
`.attrs["provenance"]`.
- Wiring: a node's input is fed by whichever of its `depends_on` produced an output of the \
SAME name. Name inputs/outputs consistently across dependent nodes so this wiring resolves. \
A connected output and input must declare the IDENTICAL contract: same `type` and, for dicts, \
the same `fields` key by key -- the producer's definition is authoritative.
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
class _PortToken:
    """Stand-in for a dependency's output value when dry-running input resolution on ports."""

    port: PortSpec


def _assignable(src: str, dst: str) -> bool:
    return src == dst or (src, dst) in {("int", "float"), ("list[int]", "list[float]")}


def _edge_field_errors(node_id: str, name: str, producer: str, src: PortSpec, dst: PortSpec) -> list[str]:
    """A consumer's declared dict keys must all be covered by the producer's declaration."""
    if dst.type != "dict" or not dst.fields:
        return []
    errors = []
    src_fields = src.fields or {}
    for key, field in dst.fields.items():
        if key not in src_fields:
            errors.append(
                f"node '{node_id}': input '{name}' requires dict key '{key}' but '{producer}' "
                f"does not declare it in its output fields"
            )
        elif not _assignable(src_fields[key].type, field.type):
            errors.append(
                f"node '{node_id}': input '{name}' key '{key}' expects {field.type} but "
                f"'{producer}' declares {src_fields[key].type}"
            )
    return errors


def _port_tokens(dag: DAGSpec) -> dict[str, dict[str, _PortToken]]:
    return {
        n.id: {name: _PortToken(port) for name, port in n.outputs.items()} for n in dag.nodes
    }


def wiring_errors(dag: DAGSpec) -> list[str]:
    """Dry-run the executor's exact input resolution with port tokens instead of values: every
    input must be fed, and fed with a compatible type -- for dicts, key by key. Catches at plan
    time the mismatches that would otherwise surface as confusing crashes inside downstream
    nodes."""
    known = {n.id for n in dag.nodes}
    tokens = _port_tokens(dag)
    errors = []
    for node in dag.nodes:
        if any(dep not in known for dep in node.depends_on):
            continue  # unknown deps are already reported by single_graph_errors
        resolved = resolve_inputs(node, {dep: tokens[dep] for dep in node.depends_on})
        for name, port in node.inputs.items():
            if name not in resolved:
                errors.append(
                    f"node '{node.id}': input '{name}' is fed by no dependency output and no param"
                )
            elif isinstance(fed := resolved[name], _PortToken):
                producer = next(
                    (dep for dep in node.depends_on if name in tokens[dep]), "a dependency"
                )
                if not _assignable(fed.port.type, port.type):
                    errors.append(
                        f"node '{node.id}': input '{name}' expects {port.type} but '{producer}' "
                        f"outputs {fed.port.type}"
                    )
                else:
                    errors += _edge_field_errors(node.id, name, producer, fed.port, port)
            elif not coarse_type_ok(fed, port.type):
                errors.append(
                    f"node '{node.id}': input '{name}' expects {port.type} but its param literal "
                    f"is {type(fed).__name__} ({fed!r})"
                )
            elif port.type == "dict":
                errors += [
                    f"node '{node.id}': param literal for {problem}"
                    for problem in dict_field_errors(f"input '{name}'", port, fed)
                ]
    return errors


def port_field_errors(dag: DAGSpec) -> list[str]:
    """Every dict output of a custom (non-registry) node must declare its exact key contract,
    and the contract must describe the value's real keys -- not wrap the value in itself."""
    errors = []
    for node in dag.nodes:
        if node.registry_id:
            continue
        for name, port in node.outputs.items():
            if port.type != "dict":
                continue
            if not port.fields:
                errors.append(
                    f"node '{node.id}': dict output '{name}' must declare `fields` (one entry per key)"
                )
            elif set(port.fields) == {name}:
                errors.append(
                    f"node '{node.id}': dict output '{name}' declares a single field also named "
                    f"'{name}' -- do not wrap the value in itself; declare the value's real keys "
                    f"(e.g. lat, lon) directly in `fields`"
                )
    return errors


def align_edge_ports(dag: DAGSpec) -> None:
    """Make every wired consumer input carry the SAME port spec as the producer's output.

    After validation the producer's declaration is authoritative: description, fields and
    example are copied over, so both sides are implemented -- and contract-tested -- against
    one identical definition instead of two independently-worded ones."""
    known = {n.id for n in dag.nodes}
    tokens = _port_tokens(dag)
    for node in dag.nodes:
        resolved = resolve_inputs(node, {dep: tokens[dep] for dep in node.depends_on if dep in known})
        for name in node.inputs:
            if isinstance(token := resolved.get(name), _PortToken):
                node.inputs[name] = token.port.model_copy(deep=True)


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
    return errors if errors else wiring_errors(dag) + registry_errors(dag) + port_field_errors(dag)


@retry_on_rate_limit
def plan(task: str, model: BaseChatModel, artifacts: RunArtifacts | None = None) -> DAGSpec:
    if artifacts:
        artifacts.save_planner_prompts(SYSTEM_PROMPT, task)
    dag, _ = run_structured_agent(model, SYSTEM_PROMPT, task, DAGSpec)
    errors = plan_errors(dag)
    if not errors:
        align_edge_ports(dag)
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
    align_edge_ports(dag)  # best effort even on an imperfect plan: one truth per edge
    return dag
