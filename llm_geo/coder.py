"""Per-node implementation: a deep agent writes a typed `run(<named inputs/params>) -> Output`
function and repairs it until its contract test (signature enforcement plus a run against
synthetic inputs, no upstream nodes involved) passes.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from .artifacts import RunArtifacts, transcript_markdown
from .contracts import ANNOTATION_BY_PORT_TYPE, run_contract
from .llm import retry_on_rate_limit
from .models import CodeEdit, NodeCodeEdits, NodeImplementation, NodeSpec, PortSpec
from .structured_output import run_structured_agent

SYSTEM_PROMPT = """You implement one DAG node as a single typed python function.

Node id: {id}
Kind: {kind}
Description: {description}

Your module must define EXACTLY this signature (same parameter names, same annotations, no
*args/**kwargs -- it is enforced):

    {signature}

where `Output` is a TypedDict you define in the same module via
`from typing_extensions import TypedDict`, with one key per declared output. Give every key the
most precise type you can commit to (e.g. `total_count: int`, `names: list[str]`,
`features: gpd.GeoDataFrame`) -- the returned value is validated against your annotation.

Inputs (passed as keyword arguments):
{inputs}

Params (static literals, passed as keyword arguments with exactly these values):
{params}

Outputs (keys of the returned dict):
{outputs}

Available libraries you may `import` inside your code: geopandas, shapely, json, math, datetime,
typing, typing_extensions.
A `GeoDataFrame` value is a geopandas.GeoDataFrame with an active geometry column and CRS set. If
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


def render_signature(node: NodeSpec) -> str:
    """The exact `run` stub the coder must implement, derived from inputs + params."""
    parts = [f"{name}: {ANNOTATION_BY_PORT_TYPE[port.type]}" for name, port in node.inputs.items()]
    parts += [
        f"{name}: {type(value).__name__}"
        for name, value in node.params.items() if name not in node.inputs
    ]
    return f"def run({', '.join(parts)}) -> Output:"


def render_ports(ports: dict[str, PortSpec]) -> str:
    lines = []
    for name, port in ports.items():
        example = f", e.g. {port.example!r}" if port.example is not None else ""
        lines.append(f"- {name} ({port.type}{example}): {port.description}")
    return "\n".join(lines) or "- (none)"


def render_params(params: dict) -> str:
    return "\n".join(f"- {name} = {value!r}" for name, value in params.items()) or "- (none)"

REPAIR_INSTRUCTIONS = """

You are REPAIRING an existing implementation: it passed its contract test but failed during real
DAG execution. Do NOT rewrite it from scratch. Return NodeCodeEdits -- a minimal list of exact
find/replace edits to the current code. Each `find` must match a unique exact substring of the
current code (including whitespace); edits are applied in order. Keep the edits as small as
possible while fixing the root cause of the execution failure. You may use the `contract_test`
tool by sending it the FULL edited code."""


def apply_edits(code: str, edits: list[CodeEdit]) -> str:
    """Apply exact find/replace edits in order; each `find` must occur exactly once."""
    if not edits:
        raise ValueError("no edits given")
    for edit in edits:
        occurrences = code.count(edit.find)
        if occurrences != 1:
            raise ValueError(
                f"edit target must occur exactly once in the current code, found {occurrences}: {edit.find[:80]!r}"
            )
        code = code.replace(edit.find, edit.replace)
    return code


def _contract_tool(node: NodeSpec):
    @tool
    def contract_test(code: str) -> str:
        """Run the candidate node code against synthetic inputs and report PASS or FAIL: <error>."""
        result = run_contract(node, code)
        return "PASS" if result.ok else f"FAIL: {result.error}"

    return contract_test


def implement_node(
    node: NodeSpec, model: BaseChatModel, max_attempts: int = 3, artifacts: RunArtifacts | None = None,
    repair_context: dict | None = None,
) -> tuple[NodeImplementation, int]:
    """Returns the implementation plus how many contract-test rounds it took to (attempt to) pass.

    When `artifacts` is given, every attempt's code, prompt, contract result and agent transcript is
    written into the run's debug bundle under nodes/<node_id>/round_RR/attempt_AA/.

    When `repair_context` is given ({"previous_code", "error", "traceback"} from a failed DAG
    execution), the node is repaired via minimal find/replace edits to the previous code instead
    of being rewritten from scratch.
    """
    system_prompt = SYSTEM_PROMPT.format(
        id=node.id, kind=node.kind.value, description=node.description,
        signature=render_signature(node),
        inputs=render_ports(node.inputs), outputs=render_ports(node.outputs),
        params=render_params(node.params),
    )
    round_no = artifacts.begin_node_round(node, system_prompt) if artifacts else 0
    tools = [_contract_tool(node)]
    invoke = retry_on_rate_limit(run_structured_agent)
    if repair_context and repair_context.get("previous_code"):
        return _repair_with_edits(
            node, invoke, model, system_prompt, tools, repair_context, max_attempts, artifacts, round_no
        )
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


def _repair_with_edits(
    node: NodeSpec, invoke, model: BaseChatModel, system_prompt: str, tools: list,
    repair_context: dict, max_attempts: int, artifacts: RunArtifacts | None, round_no: int,
) -> tuple[NodeImplementation, int]:
    code, notes, feedback = repair_context["previous_code"], "", ""
    failure = (
        f"It failed during real DAG execution with:\n{repair_context.get('error')}\n\n"
        f"Traceback:\n{repair_context.get('traceback') or '(not available)'}"
    )
    for attempt in range(1, max_attempts + 1):
        user_prompt = (
            f"Current code of node '{node.id}':\n```python\n{code}\n```\n\n{failure}\n\n"
            f"Return minimal edits that fix the root cause.{feedback}"
        )
        result, transcript = invoke(
            model, system_prompt + REPAIR_INSTRUCTIONS, user_prompt, NodeCodeEdits, tools=tools
        )
        notes = result.notes or notes
        try:
            code = apply_edits(code, result.edits)
        except ValueError as exc:
            if artifacts:
                artifacts.save_coder_attempt(
                    node.id, round_no, attempt, code=code, user_prompt=user_prompt, ok=False,
                    error=f"edits could not be applied: {exc}", notes=notes,
                    transcript_md=transcript_markdown(transcript),
                    edits_json=result.model_dump_json(indent=2),
                )
            feedback = f"\n\nYour previous edits could not be applied ({exc}); the code is unchanged. Return corrected edits."
            continue
        check = run_contract(node, code)
        if artifacts:
            artifacts.save_coder_attempt(
                node.id, round_no, attempt, code=code, user_prompt=user_prompt, ok=check.ok,
                error=check.error, notes=notes, transcript_md=transcript_markdown(transcript),
                edits_json=result.model_dump_json(indent=2),
            )
        if check.ok:
            if artifacts:
                artifacts.save_node_result(node.id, round_no, code, True, attempt)
            return NodeImplementation(node_id=node.id, code=code, notes=notes), attempt
        feedback = f"\n\nAfter your last edits the contract test fails:\n{check.error}\nReturn further edits."
    if artifacts:
        artifacts.save_node_result(node.id, round_no, code, False, max_attempts)
    return NodeImplementation(node_id=node.id, code=code, notes=notes), max_attempts
