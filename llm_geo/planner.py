"""Single-step planning: task -> validated DAGSpec, via a deep agent with structured output."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .artifacts import RunArtifacts
from .llm import retry_on_rate_limit
from .models import DAGSpec
from .registry import catalog_text
from .structured_output import run_structured_agent

SYSTEM_PROMPT = """You are a geospatial analysis planner. Given a user task, produce the COMPLETE \
workflow as a single DAG in one shot -- do not ask questions, do not explore, just plan.

Rules:
- Classify every node's `kind` as exactly one of: retrieval, transformation, synthesis.
- Every node needs a unique snake_case id, a precise `description`, `depends_on` node ids, \
`inputs`/`outputs` (name -> type, type is one of str, int, float, bool, dict, GeoDataFrame), \
and literal `params` for anything not coming from a dependency.
- Retrieval nodes must output a `GeoDataFrame`-typed value carrying provenance metadata in \
`.attrs["provenance"]`.
- Wiring: a node's input is fed by whichever of its `depends_on` produced an output of the \
SAME name. Name inputs/outputs consistently across dependent nodes so this wiring resolves.
- A trusted implementation registry is available. If a node's need is exactly covered by one \
of these, set `registry_id` to that id and make the node's inputs/outputs match it exactly. \
Otherwise leave `registry_id` null and a coding agent will implement it from your description.

Trusted registry:
""" + catalog_text() + "\n\nReturn only the final DAGSpec."


@retry_on_rate_limit
def plan(task: str, model: BaseChatModel, artifacts: RunArtifacts | None = None) -> DAGSpec:
    if artifacts:
        artifacts.save_planner_prompts(SYSTEM_PROMPT, task)
    dag, _ = run_structured_agent(model, SYSTEM_PROMPT, task, DAGSpec)
    return dag
