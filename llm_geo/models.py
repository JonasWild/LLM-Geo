"""Pydantic contracts shared across planning, implementation and execution."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    retrieval = "retrieval"
    transformation = "transformation"
    synthesis = "synthesis"


class NodeSpec(BaseModel):
    id: str = Field(description="unique snake_case node id")
    kind: NodeKind
    description: str = Field(description="what the node does, precise enough to implement")
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="input name -> type, one of str|int|float|bool|dict|GeoDataFrame or list[<one of those>]",
    )
    outputs: dict[str, str] = Field(
        default_factory=dict,
        description="output name -> type, same vocabulary as inputs",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="static literal parameters")
    registry_id: str | None = Field(
        default=None, description="id of a trusted implementation from the tool registry, if one fits"
    )
    map_over: str | None = Field(
        default=None,
        description=(
            "name of ONE list-typed input to fan out over: the trusted `registry_id` operation is "
            "run once per element of this input, with every other input/param broadcast unchanged to "
            "each call. Requires `registry_id`; the named input must be declared `list[<elem type>]` "
            "and must be an input the registry op accepts. GeoDataFrame outputs are concatenated into "
            "one GeoDataFrame; other outputs become a list of the per-element values."
        ),
    )


class DAGSpec(BaseModel):
    task: str
    nodes: list[NodeSpec]


class NodeImplementation(BaseModel):
    node_id: str
    code: str = Field(description="python source defining a function `run(**inputs) -> dict`")
    notes: str = ""


class CodeEdit(BaseModel):
    find: str = Field(description="exact substring of the current code to replace; must occur exactly once")
    replace: str = Field(description="replacement text")


class NodeCodeEdits(BaseModel):
    node_id: str
    edits: list[CodeEdit] = Field(description="minimal find/replace edits, applied in order")
    notes: str = ""


class ContractResult(BaseModel):
    ok: bool
    error: str | None = None
    output: Any = None


class ExecutionResult(BaseModel):
    success: bool
    outputs: dict[str, dict] = Field(default_factory=dict)
    failing_node_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    error_traceback: str | None = Field(
        default=None, description="full python traceback of the failing node, when the failure raised"
    )
    node_order: list[str] = Field(default_factory=list, description="nodes in the order they executed")
    node_status: dict[str, str] = Field(default_factory=dict, description="node id -> 'ok'|'error'|'cached'")
    node_duration_ms: dict[str, float] = Field(default_factory=dict)
    node_inputs: dict[str, dict] = Field(
        default_factory=dict, description="resolved inputs (params + matched upstream outputs) per executed node"
    )


class RunReport(BaseModel):
    """Everything needed to summarize one end-to-end run: plan, code sources, and outcome."""

    task: str
    dag: DAGSpec
    implementations: dict[str, NodeImplementation] = Field(default_factory=dict)
    implementation_attempts: dict[str, int] = Field(default_factory=dict, description="coder repair rounds per node")
    implement_calls: int = Field(default=0, description="total implement_one graph-node invocations (init + repairs)")
    repair_attempts: int = Field(description="assemble/execute rounds the DAG-level repair loop needed")
    result: ExecutionResult
    duration_ms: float
    agent_graph_mermaid: str = Field(default="", description="the compiled LangGraph orchestration graph")
    artifacts_dir: str = Field(default="", description="where this run's debug bundle was written")
