"""Per-node implementation: a deep agent completes a generated typed `run(...)->dict` stub and
repairs it until its contract test (run against synthetic inputs, no upstream nodes involved) passes.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from .contracts import run_contract
from .llm import retry_on_rate_limit
from .models import NodeImplementation, NodeSpec, PortSpec
from .structured_output import run_structured_agent

_PY_TYPE = {"str": "str", "int": "int", "float": "float", "bool": "bool", "dict": "dict",
            "GeoDataFrame": "gpd.GeoDataFrame"}

SYSTEM_PROMPT = """You implement one DAG node by completing EXACTLY this function -- keep the
signature verbatim (same parameter names, no *args/**kwargs):

    {signature}

Node id: {id}
Kind: {kind}
Description: {description}

Inputs:
{inputs}

Outputs (all must be present as keys of the returned dict, with exactly the declared shape):
{outputs}

Params are static literals already bound as defaulted parameters in the signature above.

Available libraries you may `import` inside your code: geopandas (as gpd), shapely, json, math,
datetime. A `GeoDataFrame` is a geopandas.GeoDataFrame with an active geometry column and CRS set;
any declared `columns`, `geometry` type, or `crs` constraint above is validated, so honor it. If
this node is a `retrieval` node, every GeoDataFrame-typed output must also carry provenance
metadata in `output_value.attrs["provenance"]` (a dict).

Common pitfalls to avoid:
- When building a GeoDataFrame manually (not via `GeoDataFrame.from_features`), always pass
  `geometry=<column>` or call `.set_geometry(...)` before any spatial method -- otherwise you get
  "the active geometry column has not been set".
- Every output value must have exactly the shape implied by its declared type, nothing looser: a
  `GeoDataFrame` output is always an actual geopandas.GeoDataFrame, never a GeoJSON dict, feature
  list, or JSON string. A downstream node consumes your output directly by its declared spec.
- Numeric outputs must be plain numbers, not strings; an `int` output must be a real int.

Use the `contract_test` tool to run your code against synthetic inputs. Iterate until it reports
PASS, then return the final NodeImplementation. Never finalize without a PASS."""


def signature_for(node: NodeSpec) -> str:
    """The exact `run` signature the implementation must define: typed inputs, then defaulted params."""
    parts = [f"{name}: {_PY_TYPE[port.type]}" for name, port in node.inputs.items()]
    parts += [f"{name}={value!r}" for name, value in node.params.items() if name not in node.inputs]
    return f"def run({', '.join(parts)}) -> dict:"


def _render_ports(ports: dict[str, PortSpec]) -> str:
    lines = []
    for name, p in ports.items():
        constraints = "; ".join(
            f"{field}={getattr(p, field)!r}" for field in ("columns", "geometry", "crs", "example")
            if getattr(p, field) is not None
        )
        line = f"- {name} ({p.type}): {p.description or 'no description'}"
        lines.append(line + (f" [{constraints}]" if constraints else ""))
    return "\n".join(lines) or "(none)"


def _contract_tool(node: NodeSpec):
    @tool
    def contract_test(code: str) -> str:
        """Run the candidate node code against synthetic inputs and report PASS or FAIL: <error>."""
        result = run_contract(node, code)
        return "PASS" if result.ok else f"FAIL: {result.error}"

    return contract_test


def implement_node(node: NodeSpec, model: BaseChatModel, max_attempts: int = 3) -> tuple[NodeImplementation, int]:
    """Returns the implementation plus how many contract-test rounds it took to (attempt to) pass."""
    system_prompt = SYSTEM_PROMPT.format(
        signature=signature_for(node), id=node.id, kind=node.kind.value, description=node.description,
        inputs=_render_ports(node.inputs), outputs=_render_ports(node.outputs),
    )
    tools = [_contract_tool(node)]
    invoke = retry_on_rate_limit(run_structured_agent)
    feedback, impl = "", None
    for attempt in range(1, max_attempts + 1):
        impl, _ = invoke(
            model, system_prompt, f"Implement node '{node.id}'.{feedback}", NodeImplementation, tools=tools
        )
        check = run_contract(node, impl.code)
        if check.ok:
            return impl, attempt
        feedback = f"\n\nYour last code:\n{impl.code}\n\nStill failing contract test:\n{check.error}\nFix it."
    return impl, max_attempts
