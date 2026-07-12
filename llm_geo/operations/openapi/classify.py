"""LLM-based classification of parsed OpenAPI operations: @code `kind` + whether the response
is GeoJSON (and should be exposed as a GeoDataFrame instead of a raw dict).

Mechanical classification isn't reliable here: `kind` and "is this GeoJSON" both depend on what
an arbitrary third-party endpoint actually does/returns semantically, not just its HTTP method or
a schema type name that's frequently just `dict[str, Any]` after $ref resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from llm_geo.llm import get_model, retry_on_rate_limit
from llm_geo.operations.openapi.models import OpenAPIDefinition

_CHUNK_SIZE = 25

_SYSTEM_PROMPT = """You classify HTTP API operations for a geospatial agentic workflow system.

For each operation, decide:
- `kind`: exactly one of
  - "retrieval": fetches/reads data from a source (typically GET), the caller mainly supplies a \
query/location/id and gets back a dataset.
  - "transformation": takes existing data (often geodata) and computes a modified/derived result \
(buffering, reprojecting, filtering, merging, symbol generation, format conversion, etc.).
  - "synthesis": computes a derived summary, statistic, or non-geodata report from inputs.
- `returns_geojson`: true only if the operation's JSON response IS (or directly wraps, e.g. one \
extra envelope field) a GeoJSON Geometry, Feature, or FeatureCollection -- i.e. it should be \
exposed to the workflow as a geopandas GeoDataFrame instead of a raw dict. False for anything
else, including responses that merely contain lat/lon numbers without GeoJSON structure.

Base your judgment on the operation's name, description, parameters, and declared response
type/description. Classify every operation given to you; do not skip any."""


@dataclass(frozen=True)
class OperationClassification:
    kind: Literal["retrieval", "transformation", "synthesis"]
    returns_geojson: bool


class _OperationClassification(BaseModel):
    function_name: str = Field(description="must exactly match one of the given operation names")
    kind: Literal["retrieval", "transformation", "synthesis"]
    returns_geojson: bool


class _ClassificationBatch(BaseModel):
    operations: list[_OperationClassification]


def _describe(definition: OpenAPIDefinition) -> str:
    params = ", ".join(f"{p.python_name}: {p.annotation}" for p in definition.parameters) or "none"
    return (
        f"- {definition.function_name} [{definition.method} {definition.path}]: {definition.description}\n"
        f"  params: {params}\n"
        f"  returns: {definition.return_annotation} -- {definition.return_description}"
    )


def _classify_chunk(
    chunk: tuple[OpenAPIDefinition, ...], model: BaseChatModel
) -> dict[str, OperationClassification]:
    listing = "\n".join(_describe(definition) for definition in chunk)
    invoke = retry_on_rate_limit(model.with_structured_output(_ClassificationBatch).invoke)
    result = invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify these {len(chunk)} operations:\n\n{listing}"},
    ])
    by_name = {item.function_name: item for item in result.operations}
    missing = [d.function_name for d in chunk if d.function_name not in by_name]
    if missing:
        raise ValueError(f"LLM classification is missing operations: {missing}")
    return {
        name: OperationClassification(kind=item.kind, returns_geojson=item.returns_geojson)
        for name, item in by_name.items()
    }


def classify_operations(
    definitions: tuple[OpenAPIDefinition, ...], model: BaseChatModel | None = None
) -> dict[str, OperationClassification]:
    """Classify every operation's @code `kind` and GeoJSON-ness, chunked into batches of
    `_CHUNK_SIZE` operations per LLM call so large OpenAPI documents stay within prompt limits."""
    if not definitions:
        return {}
    model = model or get_model()
    classifications: dict[str, OperationClassification] = {}
    for start in range(0, len(definitions), _CHUNK_SIZE):
        chunk = definitions[start : start + _CHUNK_SIZE]
        classifications.update(_classify_chunk(chunk, model))
    return classifications
