"""Pydantic contracts shared across planning, implementation and execution."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

PortType = Literal["str", "int", "float", "bool", "dict", "GeoDataFrame"]
PORT_TYPES: tuple[str, ...] = ("str", "int", "float", "bool", "dict", "GeoDataFrame")

JSONValue = str | int | float | bool | dict | list | None


FieldType = Literal[
    "str", "int", "float", "bool", "dict",
    "list[str]", "list[int]", "list[float]", "list[dict]",
]


def example_fits(type_name: str, value: Any) -> bool:
    if type_name.startswith("list["):
        inner = type_name[5:-1]
        return isinstance(value, list) and all(example_fits(inner, item) for item in value)
    match type_name:
        case "str":
            return isinstance(value, str)
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "bool":
            return isinstance(value, bool)
        case "dict":
            return isinstance(value, dict)
    return False  # GeoDataFrame: no literal example possible


class NodeKind(str, Enum):
    retrieval = "retrieval"
    transformation = "transformation"
    synthesis = "synthesis"


class FieldSpec(BaseModel):
    """The contract of one key inside a dict-typed port."""

    type: FieldType
    description: str = Field(min_length=1, description="what this key's value means")


class PortSpec(BaseModel):
    """One named input or output of a node: coarse type plus the semantics the coder needs."""

    type: PortType
    description: str = Field(
        min_length=1,
        description="what this value means: semantics, units, expected GeoDataFrame columns",
    )
    fields: dict[str, FieldSpec] | None = Field(
        default=None,
        description="dict ports only: the dict's exact contract, one entry per key",
    )
    example: JSONValue = Field(
        default=None,
        description="realistic literal sample value; scalars and dicts only, never for GeoDataFrame",
    )

    @model_validator(mode="after")
    def _drop_incoherent_extras(self) -> "PortSpec":
        # Bad optional niceties must not invalidate an otherwise good plan.
        if self.type != "dict":
            self.fields = None
        if self.example is not None and not example_fits(self.type, self.example):
            self.example = None
        if self.example is not None and self.fields and not all(
            name in self.example and example_fits(field.type, self.example[name])
            for name, field in self.fields.items()
        ):
            self.example = None  # the field contract is authoritative over a conflicting example
        return self


def _coerce_port(value: Any) -> Any:
    if isinstance(value, str):  # legacy shorthand: plain type name
        return {"type": value, "description": "(no description provided)"}
    return value


Ports = dict[str, Annotated[PortSpec, BeforeValidator(_coerce_port)]]


class NodeSpec(BaseModel):
    id: str = Field(description="unique snake_case node id")
    kind: NodeKind
    description: str = Field(description="what the node does, precise enough to implement")
    depends_on: list[str] = Field(default_factory=list)
    inputs: Ports = Field(
        default_factory=dict,
        description="input name -> port spec (type, description, optional example)",
    )
    outputs: Ports = Field(
        default_factory=dict,
        description="output name -> port spec, same shape as inputs",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="static literal parameters")
    registry_id: str | None = Field(
        default=None, description="id of a trusted implementation from the tool registry, if one fits"
    )


class DAGSpec(BaseModel):
    task: str
    nodes: list[NodeSpec]


class NodeImplementation(BaseModel):
    node_id: str
    code: str = Field(
        description="python source defining a typed function `run(<named inputs/params>) -> Output`"
    )
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
