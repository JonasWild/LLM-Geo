# llm-geo-claude

Agentic geospatial pipeline builder. Input: one natural-language task. Output: a DAG of
geospatial nodes, planned and coded by LLMs, validated, executed, reported.

## Pipeline (LangGraph state machine, `llm_geo/graph.py`)

```
plan -> implement_one (parallel, per unimplemented node) -> assemble (execute DAG)
      -> [on node failure] implement_one (repair) -> assemble -> ... (max 3 repair rounds)
```

1. **plan** (`planner.py`) — deep agent turns task string into a `DAGSpec` (structured output).
   Each `NodeSpec` = id, kind, description, depends_on, inputs, outputs, params, optional
   `registry_id`. Planner may point a node at a trusted registry impl instead of generating code.
2. **implement_one** (`coder.py`) — for each node without a `registry_id`, a deep agent writes
   `def run(**inputs) -> dict`, tests it via `contract_test` tool against synthetic inputs
   (`synthetic.py`), iterates up to 3 attempts until it passes.
3. **assemble/execute** (`executor.py`) — topologically sorts the DAG (networkx), resolves each
   node's inputs from upstream outputs by matching name, falling back to positional pairing,
   runs each node's callable in order, stops at first failure.
4. **repair loop**: if execution fails, re-invoke `implement_one` only for the failing
   LLM-generated node(s), re-assemble. Up to `MAX_REPAIR_ATTEMPTS=3` (graph.py:16).
5. Result packaged as `RunReport` (`models.py`): dag, implementations, per-node coder attempts,
   ExecutionResult (success, outputs, node_order/status/duration, error).

## Contracts (`contracts.py`)

Node code is `exec`'d, must define `run`. Contract check: call `run(**synthetic_inputs)`, verify
return is a dict with all declared `outputs` present; `geojson`-typed outputs must be a
`{"type": "FeatureCollection", "features": [...]}` dict; `retrieval`-kind geojson outputs must
also carry `provenance`. This check runs standalone (synthetic inputs), never against real
upstream data — passing it does not guarantee runtime success against real data, hence the
DAG-level repair loop in step 4.

## Type vocabulary

Input/output types (string values in `NodeSpec.inputs`/`outputs`): `str`, `int`, `float`, `bool`,
`dict`, `GeoDataFrame`. `GeoDataFrame` = real `geopandas.GeoDataFrame`, not a GeoJSON dict.
Retrieval-kind `GeoDataFrame` outputs must carry provenance metadata in `.attrs["provenance"]`.

## Trusted operations (ground truth: `llm_geo/operations/registry.py`)

The `@code(kind=...)` decorator in `llm_geo/operations/registry.py` registers a top-level,
fully-typed Python function as a trusted operation: it inspects real type hints (so the type
vocabulary above is derived, not hand-declared) and requires a Google-style docstring with
`Args:`/`Returns:` sections matching the signature exactly. Registered functions take plain typed
params (no `*args`/`**kwargs`/keyword-only) and return one concrete value (not a named-output
dict). `registered_operations()` returns everything registered, sorted by id.

`llm_geo/tools/public_data_providers.py` is where the actual trusted operations live, decorated
with `@code`:

| id | kind | inputs | returns |
|---|---|---|---|
| `read_geojson` | retrieval | path:str | GeoDataFrame |
| `geocode_place` | retrieval | query:str, limit:int=5 | GeoDataFrame |
| `overpass_query` | retrieval | amenity:str, bbox:str, limit:int=20 | GeoDataFrame |
| `buffer` | transformation | features:GeoDataFrame, distance:float | GeoDataFrame |
| `reproject` | transformation | features:GeoDataFrame, crs:str | GeoDataFrame |
| `filter_intersects` | transformation | features:GeoDataFrame, mask_features:GeoDataFrame | GeoDataFrame |
| `summarize` | synthesis | features:GeoDataFrame | dict |

`geocode_place` hits Nominatim, `overpass_query` hits Overpass API — both live network calls,
retried 3x on 5xx.

`llm_geo/registry.py` is an **adapter**, not ground truth: it imports `public_data_providers`
(triggering `@code` registration as a side effect), then wraps each `RegisteredOperation`'s single
return value into the `{name: value}` dict shape the planner/executor/coder already use —
`GeoDataFrame`-returning ops become `{"features": <GeoDataFrame>}`, `dict`-returning ops become
`{"report": {...}}` (see `_OUTPUT_NAME_BY_TYPE`). This is what `planner.py`/`executor.py` actually
import (`REGISTRY`, `catalog_text`) — neither had to change to support the new ground truth.

Note: `llm_geo/operations/` also contains an unrelated, not-yet-wired-into-the-pipeline subsystem
for generating trusted operations from OpenAPI specs (`llm_geo/operations/openapi/`,
`llm_geo/operations/generated/`, `generate_openapi_operations.py`), plus FAISS/TF-IDF hybrid
retrieval (`retrieval.py`) for large catalogs (needs `faiss`/`scikit-learn`, not in `pyproject.toml`).

### OpenAPI-generated operations

`llm_geo/operations/generate_openapi_operations.py` fetches each configured server's OpenAPI spec
(`openapi/servers.py`), parses it (`openapi/parser.py`), and renders one `@code`-decorated function
per endpoint (`openapi/renderer.py`) into `llm_geo/operations/generated/<service>.py`, validated by
importing the generated module in a subprocess before atomically promoting it.

Since `@code` requires a `kind` and neither HTTP method nor OpenAPI schema reliably says whether a
third-party endpoint's JSON response is GeoJSON, `openapi/classify.py` classifies every parsed
operation with an LLM call (chunked at 25 ops/call): its `kind` (retrieval/transformation/synthesis)
and `returns_geojson`. For `returns_geojson` retrieval operations, the renderer swaps the return
annotation to `GeoDataFrame` and emits a call to `openapi/runtime.py`'s `geojson_to_geodataframe`
(handles FeatureCollection/Feature/bare-Geometry payloads, stamps `.attrs["provenance"]`) instead of
returning the raw decoded JSON dict — matching the `GeoDataFrame`-for-geodata convention everywhere
else in the type system. Classification only runs when a service's spec actually changes (or
`force=True`); results are recorded in that service's `generated/specs/<service>.manifest.json`.

## Files

| File | Role |
|---|---|
| `llm_geo/graph.py` | LangGraph orchestration: state machine wiring plan/implement/assemble/repair |
| `llm_geo/planner.py` | task string -> `DAGSpec` |
| `llm_geo/coder.py` | `NodeSpec` -> `NodeImplementation` (generated code), self-repairing via contract_test |
| `llm_geo/contracts.py` | compile + run node code against synthetic inputs, validate output shape |
| `llm_geo/synthetic.py` | fake inputs per declared type, for contract testing without upstream data |
| `llm_geo/executor.py` | topological execution of the real DAG against real data |
| `llm_geo/registry.py` | adapter: ground-truth operations -> the id->{kind,inputs,outputs,fn} shape the pipeline expects |
| `llm_geo/operations/registry.py` | ground truth: `@code` decorator + `RegisteredOperation` registration mechanism |
| `llm_geo/tools/public_data_providers.py` | the actual trusted operations (`@code`-decorated), incl. live OSM (Nominatim/Overpass) calls |
| `llm_geo/models.py` | shared pydantic models: NodeSpec, DAGSpec, NodeImplementation, ExecutionResult, RunReport |
| `llm_geo/llm.py` | `ChatOpenAI`/`OpenAIEmbeddings` factories (model/base URL/API key configurable via env, see `.env.example`), rate-limit retry, deepagents harness profile (strips unused filesystem/shell tools) |
| `llm_geo/trace.py` | `Tracer`: JSONL event log to `traces/run.jsonl` + telegraphic stdout logging |
| `llm_geo/report.py` | `RunReport` -> Mermaid solution/execution graphs + Markdown report |
| `llm_geo/cli.py` | `python -m llm_geo.cli "<task>"` single-task entrypoint |
| `main.py` | runs a fixed suite of ~20 end-to-end test tasks, writes `reports/run_<timestamp>.md` |
| `tests/test_pipeline.py` | offline pytest suite (no LLM calls): contracts, executor, registry pipeline, cycle detection |
| `data/*.geojson` | sample fixtures used by `main.py` test cases |

## Run

```
poetry install
cp .env.example .env                    # fill in OPENAI_API_KEY at minimum
python -m llm_geo.cli "Buffer sample points by 100m and summarize."
python main.py                          # full test-case suite -> reports/run_*.md
pytest                                   # offline unit tests, no API key needed
```

Env vars (see `.env.example`): `OPENAI_API_KEY` (required), `OPENAI_MODEL` (optional, default
`gpt-4o-mini`), `OPENAI_BASE_URL` (optional, point at a custom OpenAI-compatible endpoint),
`OPENAI_EMBEDDING_MODEL` (optional, default `text-embedding-3-small`), and
`OPENAI_EMBEDDING_API_KEY`/`OPENAI_EMBEDDING_BASE_URL` (optional, override the shared LLM
key/base URL for embeddings specifically).

## Key invariants for editing this code

- A node's `run(**inputs) -> dict` must return exactly its declared `outputs` keys, no more/less
  nesting than the declared type implies. This is the LLM-generated custom-node convention
  (`coder.py`); trusted `@code` operations instead return one bare value and get wrapped into that
  shape by the `llm_geo/registry.py` adapter.
- `_resolve_inputs` (executor.py:21) matches inputs to dependency outputs by **name first**, then
  falls back to positional pairing over remaining unresolved input/dependency pairs — keep planner
  output naming consistent across dependent nodes or wiring silently breaks.
- Registry nodes never go through `coder.py`/contract testing — they're trusted as-is.
- New trusted operations belong in `llm_geo/tools/public_data_providers.py` (or a sibling module),
  decorated with `@code(kind=...)`: module-scope function, concrete type hints on every param and
  the return, no `*args`/`**kwargs`/keyword-only params, Google-style docstring with `Args:`
  (every param, exactly) and `Returns:` sections. `llm_geo/registry.py` picks it up automatically
  via `registered_operations()` — no adapter changes needed unless the output_type isn't already
  in `_OUTPUT_NAME_BY_TYPE`.
- `implement_node`/`execute` both raise on hard failures; `graph.py` catches only what
  `ExecutionResult`/repair routing expects — see `route_after_assemble` (graph.py:59).
