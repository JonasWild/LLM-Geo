"""Pydantic contracts shared across planning, implementation and execution."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class NodeKind(str, Enum):
    retrieval = "retrieval"
    transformation = "transformation"
    synthesis = "synthesis"


PortType = Literal["str", "int", "float", "bool", "dict", "GeoDataFrame"]


class PortSpec(BaseModel):
    """Full contract of one named input/output value of a node."""

    type: PortType
    description: str = Field(default="", description="semantic meaning of the value, incl. units")
    columns: dict[str, str] | None = Field(
        default=None, description="GeoDataFrame only: required column name -> dtype (str|int|float|bool)"
    )
    geometry: Literal["Point", "LineString", "Polygon", "any"] | None = Field(
        default=None, description="GeoDataFrame only: expected geometry type"
    )
    crs: str | None = Field(default=None, description="GeoDataFrame only: expected CRS, e.g. 'EPSG:4326'")
    example: Any | None = Field(default=None, description="realistic example value (non-GeoDataFrame ports)")


def _coerce_ports(value: Any) -> Any:
    """Accept the legacy `name -> type-string` shorthand and lift it into PortSpecs."""
    if isinstance(value, dict):
        return {k: {"type": v} if isinstance(v, str) else v for k, v in value.items()}
    return value


class NodeSpec(BaseModel):
    id: str = Field(description="unique snake_case node id")
    kind: NodeKind
    description: str = Field(description="what the node does, precise enough to implement")
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, PortSpec] = Field(default_factory=dict, description="input name -> port spec")
    outputs: dict[str, PortSpec] = Field(default_factory=dict, description="output name -> port spec")

    _ports = field_validator("inputs", "outputs", mode="before")(_coerce_ports)
    params: dict[str, Any] = Field(default_factory=dict, description="static literal parameters")
    registry_id: str | None = Field(
        default=None, description="id of a trusted implementation from the tool registry, if one fits"
    )


class DAGSpec(BaseModel):
    task: str
    nodes: list[NodeSpec]


class NodeImplementation(BaseModel):
    node_id: str
    code: str = Field(description="python source defining a function `run(**inputs) -> dict`")
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
    node_order: list[str] = Field(default_factory=list, description="nodes in the order they executed")
    node_status: dict[str, str] = Field(default_factory=dict, description="node id -> 'ok'|'error'")
    node_duration_ms: dict[str, float] = Field(default_factory=dict)


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
