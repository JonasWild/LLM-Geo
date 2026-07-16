"""Single-step planning: task -> validated DAGSpec, via a deep agent with structured output."""
from __future__ import annotations

import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel

from .artifacts import RunArtifacts
from .llm import retry_on_rate_limit
from .models import DAGSpec
from .registry import catalog_text
from .structured_output import run_structured_agent

SYSTEM_PROMPT = ("""You are a geospatial analysis planner. Given a user task, produce the COMPLETE \
workflow as a single DAG in one shot -- do not ask questions, do not explore, just plan.

Rules:
- Classify every node's `kind` as exactly one of: retrieval, transformation, synthesis.
- Every node needs a unique snake_case id, a precise `description`, `depends_on` node ids, \
`inputs`/`outputs` (name -> type, type is one of str, int, float, bool, dict, GeoDataFrame), \
and literal `params` for anything not coming from a dependency.
- The nodes must form exactly ONE connected workflow graph: every node must be linked to the \
rest via `depends_on` (no disconnected islands, no parallel unrelated graphs), converging on \
the final answer node(s) for the task.
- Retrieval nodes must output a `GeoDataFrame`-typed value carrying provenance metadata in \
`.attrs["provenance"]`.
- Wiring: a node's input is fed by whichever of its `depends_on` produced an output of the \
SAME name. Name inputs/outputs consistently across dependent nodes so this wiring resolves.
- A trusted implementation registry is available. If a node's need is exactly covered by one \
of these, set `registry_id` to that id and make the node's inputs/outputs match it exactly. \
Otherwise leave `registry_id` null and a coding agent will implement it from your description.

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


@retry_on_rate_limit
def plan(task: str, model: BaseChatModel, artifacts: RunArtifacts | None = None) -> DAGSpec:
    if artifacts:
        artifacts.save_planner_prompts(SYSTEM_PROMPT, task)
    dag, _ = run_structured_agent(model, SYSTEM_PROMPT, task, DAGSpec)
    errors = single_graph_errors(dag)
    if not errors:
        return dag

    # One corrective round: show the planner its own invalid plan plus the specific violations.
    retry_prompt = (
        f"{task}\n\nYour previous plan was invalid:\n- " + "\n- ".join(errors)
        + f"\n\nPrevious plan:\n{dag.model_dump_json()}\n\n"
        "Return a corrected DAGSpec whose nodes form exactly ONE connected workflow graph."
    )
    if artifacts:
        artifacts.record_error("plan_validation", "planner produced an invalid DAG, retrying once:\n- " + "\n- ".join(errors))
        artifacts.write("plan/prompts/user_retry.md", retry_prompt)
    dag, _ = run_structured_agent(model, SYSTEM_PROMPT, retry_prompt, DAGSpec)
    remaining = single_graph_errors(dag)
    if remaining and artifacts:
        artifacts.record_error(
            "plan_validation", "planner DAG is still invalid after one retry, proceeding anyway:\n- " + "\n- ".join(remaining)
        )
    return dag
