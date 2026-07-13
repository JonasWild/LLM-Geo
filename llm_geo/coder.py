"""Per-node implementation: a deep agent writes `run(**inputs)->dict` and repairs it until its
contract test (run against synthetic inputs, no upstream nodes involved) passes.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from .artifacts import RunArtifacts, transcript_markdown
from .contracts import run_contract
from .llm import retry_on_rate_limit
from .models import NodeImplementation, NodeSpec
from .structured_output import run_structured_agent

SYSTEM_PROMPT = """You implement one DAG node as a single python function:

    def run(**inputs) -> dict: ...

Node id: {id}
Kind: {kind}
Description: {description}
Inputs (name: type): {inputs}
Outputs (name: type, must all be present as dict keys returned by run): {outputs}
Params (static literals also available as kwargs): {params}

Available libraries you may `import` inside your code: geopandas, shapely, json, math, datetime.
A `GeoDataFrame` type is a geopandas.GeoDataFrame with an active geometry column and CRS set. If
this node is a `retrieval` node, every GeoDataFrame-typed output must also carry provenance
metadata in `output_value.attrs["provenance"]` (a dict).

Common pitfalls to avoid:
- When building a GeoDataFrame manually (not via `GeoDataFrame.from_features`), always pass
  `geometry=<column>` or call `.set_geometry(...)` before any spatial method -- otherwise you get
  "the active geometry column has not been set".
- Every output value must have exactly the shape implied by its declared type, nothing looser: a
  `GeoDataFrame` output is always an actual geopandas.GeoDataFrame, never a GeoJSON dict, feature
  list, or JSON string. A downstream node may consume your output directly by its declared type,
  so do not invent extra nesting or return a differently-shaped value even if it "also passes"
  your own contract test.
- Numeric outputs (type `float`/`int`) must be plain numbers, not strings.

Use the `contract_test` tool to run your code against synthetic inputs. Iterate until it reports
PASS, then return the final NodeImplementation. Never finalize without a PASS."""


def _contract_tool(node: NodeSpec):
    @tool
    def contract_test(code: str) -> str:
        """Run the candidate node code against synthetic inputs and report PASS or FAIL: <error>."""
        result = run_contract(node, code)
        return "PASS" if result.ok else f"FAIL: {result.error}"

    return contract_test


def implement_node(
    node: NodeSpec, model: BaseChatModel, max_attempts: int = 3, artifacts: RunArtifacts | None = None
) -> tuple[NodeImplementation, int]:
    """Returns the implementation plus how many contract-test rounds it took to (attempt to) pass.

    When `artifacts` is given, every attempt's code, prompt, contract result and agent transcript is
    written into the run's debug bundle under nodes/<node_id>/round_RR/attempt_AA/.
    """
    system_prompt = SYSTEM_PROMPT.format(
        id=node.id, kind=node.kind.value, description=node.description,
        inputs=node.inputs, outputs=node.outputs, params=node.params,
    )
    round_no = artifacts.begin_node_round(node, system_prompt) if artifacts else 0
    tools = [_contract_tool(node)]
    invoke = retry_on_rate_limit(run_structured_agent)
    feedback, impl = "", None
    for attempt in range(1, max_attempts + 1):
        user_prompt = f"Implement node '{node.id}'.{feedback}"
        impl, transcript = invoke(model, system_prompt, user_prompt, NodeImplementation, tools=tools)
        check = run_contract(node, impl.code)
        if artifacts:
            artifacts.save_coder_attempt(
                node.id, round_no, attempt, code=impl.code, user_prompt=user_prompt,
                ok=check.ok, error=check.error, notes=impl.notes,
                transcript_md=transcript_markdown(transcript),
            )
        if check.ok:
            if artifacts:
                artifacts.save_node_result(node.id, round_no, impl.code, True, attempt)
            return impl, attempt
        feedback = f"\n\nYour last code:\n{impl.code}\n\nStill failing contract test:\n{check.error}\nFix it."
    if artifacts:
        artifacts.save_node_result(node.id, round_no, impl.code, False, max_attempts)
    return impl, max_attempts
