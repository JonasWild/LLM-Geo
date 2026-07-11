# LLM-GEO

Autonomous GIS. Deep Agents supervision. LangGraph execution.

## Configure and run

Install the project and copy the environment template:

```powershell
poetry install --no-root
Copy-Item .env.example .env
```

Configure the model and credentials in `.env`:

```dotenv
LLM_GEO_MODEL=gpt-5.4-mini
LLM_GEO_MODEL_PROVIDER=openai
OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=
```

`OPENAI_BASE_URL` is optional for OpenAI itself. To use an OpenAI Chat
Completions-compatible server such as LM Studio, vLLM, or an OpenAI-compatible
Ollama endpoint, set its `/v1` URL and change only the model name and API key as
required by that server. A custom URL explicitly uses the Chat Completions API:

```dotenv
LLM_GEO_MODEL=my-local-model
LLM_GEO_MODEL_PROVIDER=openai
OPENAI_API_KEY=not-needed
OPENAI_BASE_URL=http://localhost:1234/v1
```

The remaining durable runtime settings are documented in `.env.example`:
`LLM_GEO_USE_DEEP_AGENT`,
`LLM_GEO_ALLOW_CODE_EXECUTION`, `LLM_GEO_OUTPUT_ROOT`, retry limits,
`LLM_GEO_LOG_LEVEL`, `LLM_GEO_GENERATE_MERMAID`, and
`LLM_GEO_SLOW_STEP_SECONDS`. Values already set in the shell take precedence over
`.env`.

Outgoing model, Overpass, and Nominatim HTTP request logging is enabled by default:

```dotenv
LLM_GEO_LOG_HTTP=true
```

This records request metadata such as method, URL, and response status in the run's
`llm_geo.log` without deliberately logging headers or bodies. HTTPX model-request
summaries appear on the console at `INFO`. Set `LLM_GEO_LOG_LEVEL=DEBUG` when you also
want the lower-level `requests`/urllib3 activity on the console. Debug URLs can contain
query parameters, so use that setting only where geographic queries may safely appear.
Set `LLM_GEO_LOG_HTTP=false` to disable all HTTP client request logging.
Set `LLM_GEO_GENERATE_MERMAID=false` to skip both Mermaid source and PNG workflow
diagnostics. Structured execution tracing remains enabled.
Important internal operations exceeding `LLM_GEO_SLOW_STEP_SECONDS` (default: 10)
produce a concise `Slow step` warning with their measured duration.

Run the default task from `main.py`, or override the per-run values on the command
line:

```powershell
poetry run python main.py
poetry run python main.py --task "Map parks in Berlin" --task-name berlin_parks
```

An empty task or `LLM_GEO_MODEL` performs a readiness check without an LLM request.
Retrieval tool registration and trusted Python operation registration remain in
`main.py`, because they are Python objects rather than scalar configuration.

Set `LLM_GEO_USE_DEEP_AGENT=true` to route the task through the conversational Deep
Agents supervisor. The supervisor exposes the complete workflow as one
`run_geospatial_analysis` tool and forwards the same output, execution, retry,
and logging settings as the direct entry path.

`PUBLIC_RETRIEVAL_OPERATIONS` registers the built-in `@code` operations
`overpass_to_geojson` and `nominatim_to_geojson`. The planner places retrieval in the
same DAG as every other operation. Both functions persist a GeoJSON FeatureCollection
and return its features as an EPSG:4326 GeoDataFrame.
For Nominatim, set an identifying user agent with contact information in `.env`:

```dotenv
NOMINATIM_USER_AGENT=LLM-GEO/0.2 (contact: you@example.com)
OVERPASS_URL=
NOMINATIM_URL=
```

The Nominatim tool permits one request per second and limits each search to 50 results.
Overpass queries are supplied as Overpass QL and use the public interpreter endpoint;
keep them spatially bounded and narrowly scoped. Set `OVERPASS_URL` to a complete
Overpass interpreter URL or `NOMINATIM_URL` to a complete Nominatim search URL to use
a self-hosted instance or proxy. When `OVERPASS_URL` is set, requests stay on that
endpoint instead of falling back to the public mirror pool.

## Registered Operations

Prewritten, trusted Python functions can be selected by the workflow planner without
being rewritten by an LLM. Define a top-level function in an importable module and
decorate it with zero-argument `@code`. The qualified function name is its registry ID;
parameter annotations, return annotation, and the docstring become its capability
contract.

When more operations are registered than `LLM_GEO_OPERATION_RETRIEVAL_LIMIT`, the
planner catalog is selected from complete operation docstrings using local hybrid
FAISS and lexical retrieval. The full trusted registry remains available to DAG
validation and execution. Embeddings use OpenAI by default:

```dotenv
LLM_GEO_OPERATION_RETRIEVAL=true
LLM_GEO_OPERATION_RETRIEVAL_LIMIT=50
LLM_GEO_EMBEDDING_MODEL=text-embedding-3-small
```

For a locally deployed OpenAI-compatible embeddings endpoint, configure only its
embedding URL and model (plus a placeholder key if the server requires one):

```dotenv
LLM_GEO_EMBEDDING_MODEL=bge-m3
LLM_GEO_EMBEDDING_BASE_URL=http://localhost:8000/v1
LLM_GEO_EMBEDDING_API_KEY=not-needed
```

The FAISS index is cached under `.llm_geo/operation_index` and rebuilt when an
operation docstring, operation ID, embedding model, or embedding endpoint changes.

```python
import geopandas as gpd

from llm_geo.operations import code


@code
def filter_named_places(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
  """Keep features that have a name.

  Args:
    features: Inspected GeoJSON features with OSM attributes.

  Returns:
    Features whose name attribute is populated.
  """
  return features.loc[features["name"].notna()].copy()
```

Operations default to the `transformation` category. Data-acquisition operations
should use `@code(category="retrieval")`. Retrieval operations start a DAG branch,
accept no incoming graph data, and receive task-known queries, paths, and other
configuration through `literal_arguments`; their return value produces the first
data node in that branch.

Import the module before calling `registered_operations()` in `main.py`, then pass the
returned tuple as `registered_operations=REGISTERED_OPERATIONS`. Registered functions
must be module-level, fully annotated, use no `*args`, `**kwargs`, or keyword-only
arguments, return one concrete value, and include `Args:` and `Returns:` docstring
sections. The first implementation supports one graph output per registered function;
use generated operations for multi-output steps.

### Generated OpenAPI operations

Reviewed services in `llm_geo/operations/openapi/servers.py` can be converted directly
from their OpenAPI JSON documents into the same strict `@code` functions. Generation
does not run during agent startup and does not require `openapi-python-client`. Run the
hard-coded synchronizer explicitly during development:

```powershell
poetry run python -m llm_geo.operations.generate_openapi_operations
```

Each entry in `OPENAPI_SERVERS` selects exactly one schema source. A URL can be used:

```python
{
    "service": "geo_mcp",
    "openapi_url": "http://schema-host:8000/openapi.json",
    "base_url": "http://operation-host:8000",
}
```

Or the schema can be read from a local file. Relative paths are resolved from the
repository root:

```python
{
    "service": "geo_mcp",
    "openapi_path": "openapi/geo_mcp.json",
    "base_url": "http://operation-host:8000",
}
```

`base_url` is deliberately separate from the schema source and is embedded as the
default endpoint for generated functions. `LLM_GEO_OPENAPI_GEO_MCP_URL` can still
override it at runtime without regenerating the module.

It normalizes and snapshots each schema, renders a module under
`llm_geo/operations/generated/`, imports the candidate in an isolated interpreter,
checks that every expected function registers, and only then replaces the previous
module. Unsupported endpoints such as binary downloads are recorded with reasons in
the generated manifest. Unchanged schema hashes are skipped.

Generated operations are imported through the same reviewed server allowlist before
`registered_operations()` is evaluated. Runtime configuration remains outside the
generated source:

```dotenv
LLM_GEO_OPENAPI_GEO_MCP_URL=http://localhost:8000
LLM_GEO_OPENAPI_GEO_MCP_API_KEY=
LLM_GEO_OPENAPI_TIMEOUT=30
```

For local models that support JSON mode but do not reliably call LangChain's
synthetic structured-output tool, select application-side Pydantic parsing:

```dotenv
LLM_GEO_STRUCTURED_OUTPUT=json_mode
```

The supported values are `auto` (LangChain's default selection), `tool` (an explicit
synthetic result tool), `provider` (provider-native JSON Schema), `json_mode`
(`response_format=json_object` plus Pydantic validation), and `prompted` (schema
instructions in the prompt plus manual JSON extraction and Pydantic validation).
JSON and prompted modes use the same structured planning response as other modes.

The generator currently supports path, query, header, and JSON request inputs plus
JSON `2xx` responses. Multipart bodies, binary responses, callbacks, and external
schema references are skipped rather than guessed.

## System

`llm_geo/system.py` is the execution center:

```text
task + registered operations
  → generate one typed retrieval/analysis data-operation DAG
  → validate DAG; replan if invalid
  → generate and review operations
  → assemble and review program
  → execute in subprocess
  → repair from traceback; retry
  → validate manifest and semantics
  → persist artifacts and checkpoints
```

Retrieval operations are ordinary trusted `@code` functions. Task values such as an
Overpass query, output path, or result limit are stored in the operation node's
`literal_arguments`; data edges bind the remaining function parameters. Retrieval
operations may therefore start the DAG without an input data node. Provider
credentials remain environment configuration.

## Navigation

```text
main.py                         task configuration + executable entry
llm_geo/
  system.py                     main LangGraph; run/resume orchestration
  subagents/
    runtime.py                  structured agent calls; code review
    supervisor.py               Deep Agents supervisor
  tools/
    public_data_providers.py     Overpass/Nominatim @code retrieval operations
    data_inspection.py          table, vector, raster inspection
    workflow_graph.py           DAG validation; GraphML/PNG/HTML
    code_execution.py           subprocess execution; artifacts
  operations/
    registry.py                 @code registration for trusted functions
    openapi/                    direct OpenAPI parser, renderer, runtime, generator
    generated/                  allowlisted generated @code modules and spec snapshots
  middleware/
    logging.py                  concise console + detailed file logs
  utils/
    models.py                   typed contracts and graph state
    prompts.py                  shared GIS policy
```

Public API:

```python
from llm_geo import (
    create_geo_agent,
    create_llm_geo_graph,
    resume_llm_geo,
    run_llm_geo,
)
```

## Output

One isolated directory per run:

```text
output/<task_name>/<UTC timestamp>/
  llm_geo.log
  <task_name>.checkpoints.sqlite
  <task_name>.state.json
  prompts/
    001_plan_01.txt
    002_code_<operation>_01.txt
    003_review_<operation>_01.txt
    ...                           # only calls that actually occurred
  data/                           reserved run data
  workflow/
    plan.json
    graph.graphml
    graph.html
    graph.png
    system.mmd                     # every possible top-level agent route
    system.png
    execution.mmd                  # route actually taken, with retries/timings
    execution.png
  code/
    solution.py                    # latest candidate; validated solution on success
    revisions.jsonl                # chronological code-revision metadata
    revisions/
      001_assembler_raw.py
      002_assembler_reviewed.py
      003_debugger.py              # repairs are never overwritten
      ...
    executions/
      attempt_001.py               # exact source submitted to the subprocess
      attempt_001.json             # outcome, stdout/stderr, timing, revision
      ...
  results/
    llm_geo_result.json
    ...maps, charts, reports
```

`code/revisions/` traces every complete program produced by assembly, review,
debugging, or validation. `code/executions/` separately records exactly which source
was run and whether it succeeded, failed, timed out, or was skipped because execution
was disabled. Revision numbering continues when a checkpointed run is resumed.
`code/solution.py` remains the stable entry point: it contains the validated program
after success and the latest attempted candidate after terminal failure.

Resume: `resume_llm_geo(model, run_dir="output/<task>/<timestamp>")`.

The Mermaid PNGs are rendered locally. A global `mmdc` is used when available;
otherwise Node.js/npm runs the pinned Mermaid CLI through `npx` and caches it. If no
renderer is available, the inspectable `.mmd` source is retained and the GIS run
continues with a warning.

## Logs

- Console: stage, progress, retries, result, failure.
- File: console events plus DEBUG detail.
- `workflow/execution.png`: chronological top-level stage log, including retries,
  outcomes, and durations.
- Prompt contents are not copied into `llm_geo.log`. The `prompts/` artifacts
  intentionally preserve submitted prompts and can contain generated code,
  tracebacks, result metadata, and local paths; handle the run directory accordingly.
- Never persisted or logged: provider credentials and environment secrets.

## Tests

Run the deterministic unit tests and the offline DeepEval checks separately:

```powershell
poetry run python -m unittest discover -s tests/unit -v
$env:DEEPEVAL_DISABLE_DOTENV = "1"
$env:DEEPEVAL_TELEMETRY_OPT_OUT = "1"
poetry run python -m pytest tests/evals -q
```

The DeepEval checks use deterministic local metrics only. They do not call an LLM,
public data provider, or evaluation service.
