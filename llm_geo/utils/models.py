"""Typed state and agent contracts."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class DataSource(BaseModel):
    description: str = Field(min_length=1)
    location: str = Field(min_length=1)
    format: Literal["GeoJSON"] = "GeoJSON"
    provider: str = Field(min_length=1)
    request: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    inspection_error: str | None = None


class PlanNode(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    kind: Literal["data", "operation"]
    description: str = Field(min_length=1)
    data_path: str = ""
    implementation: Literal["generated", "registered"]
    registered_operation_id: str | None = None
    literal_arguments: dict[str, Any] = Field(default_factory=dict)
    generation_reason: str | None = None

    @model_validator(mode="after")
    def validate_implementation_metadata(self) -> "PlanNode":
        if self.kind == "data" and (
            self.implementation != "generated"
            or self.registered_operation_id
            or self.literal_arguments
            or self.generation_reason
        ):
            raise ValueError(
                "Data nodes cannot select an implementation or registered operation."
            )
        if self.kind == "operation" and self.implementation == "generated":
            if not self.generation_reason or not self.generation_reason.strip():
                raise ValueError(
                    "Generated operations must explain why no registered operation applies."
                )
        if self.implementation == "registered" and self.generation_reason:
            raise ValueError(
                "Registered operations cannot include a generation reason."
            )
        return self


class PlanEdge(BaseModel):
    source: str
    target: str


class WorkflowPlan(BaseModel):
    rationale: str
    nodes: list[PlanNode] = Field(min_length=2)
    edges: list[PlanEdge] = Field(min_length=1)


class CodeArtifact(BaseModel):
    code: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    corrected_code: str | None = None


class ResultValidation(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    corrected_code: str | None = None


class WorkflowStep(BaseModel):
    node_id: str
    description: str
    inputs: list[str]
    outputs: list[str]
    code: str
    review_issues: list[str] = Field(default_factory=list)
    registered_operation_id: str | None = None


class LLMGeoState(TypedDict, total=False):
    task: str
    task_name: str
    save_dir: str
    allow_code_execution: bool
    max_plan_attempts: int
    max_execution_attempts: int
    plan_attempts: int
    execution_attempts: int
    plan: dict[str, Any]
    plan_issues: list[str]
    retrieved_operation_ids: list[str]
    operations: list[dict[str, Any]]
    assembled_code: str
    execution: dict[str, Any]
    validation: dict[str, Any]
    error: str
    status: str
    artifacts: list[str]
    execution_trace: Annotated[list[dict[str, Any]], operator.add]
    checkpoint_thread_id: str
