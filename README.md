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
`LLM_GEO_DIRECT_MODE`, `LLM_GEO_USE_DEEP_AGENT`,
`LLM_GEO_ALLOW_CODE_EXECUTION`, `LLM_GEO_OUTPUT_ROOT`, retry limits, and
`LLM_GEO_LOG_LEVEL`. Values already set in the shell take precedence over `.env`.

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
and logging settings as the direct entry path. `LLM_GEO_DIRECT_MODE` independently
controls graph decomposition inside that workflow.

`PUBLIC_RETRIEVAL_TOOLS` registers the built-in `overpass_to_geojson` and
`nominatim_to_geojson` tools. Both write local GeoJSON FeatureCollections only.
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

Import the module before calling `registered_operations()` in `main.py`, then pass the
returned tuple as `registered_operations=REGISTERED_OPERATIONS`. Registered functions
must be module-level, fully annotated, use no `*args`, `**kwargs`, or keyword-only
arguments, return one concrete value, and include `Args:` and `Returns:` docstring
sections. The first implementation supports one graph output per registered function;
use generated operations for multi-output steps.

## System

`llm_geo/system.py` is the execution center:

```text
task + registered retrieval providers
  → retrieve validated GeoJSON and inspect sources
  → generate typed data/operation DAG
  → validate DAG; replan if invalid
  → generate and review operations
  → assemble and review program
  → execute in subprocess
  → repair from traceback; retry
  → validate manifest and semantics
  → persist artifacts and checkpoints
```

Every provider tool must materialize a GeoJSON FeatureCollection inside the current
run's `data/` directory and return its local path, provider name, description, and
request/provenance metadata as TOON. LLM-facing structured context and tool messages
use TOON to reduce token usage; persisted GeoJSON, manifests, and workflow state remain
JSON. The workflow rejects URLs, paths outside that directory, missing files, invalid
TOON metadata, and non-FeatureCollection GeoJSON before planning. Provider credentials
belong in environment variables and are never stored in source metadata or logs.

Direct mode skips DAG decomposition; retains retrieval, inspection, review, execution,
repair, and validation.

## Navigation

```text
main.py                         task configuration + executable entry
llm_geo/
  system.py                     main LangGraph; run/resume orchestration
  subagents/
    runtime.py                  structured agent calls; code review
    supervisor.py               Deep Agents supervisor
  tools/
    public_data_providers.py     Overpass/Nominatim GeoJSON retrieval tools
    data_inspection.py          table, vector, raster inspection
    workflow_graph.py           DAG validation; GraphML/PNG/HTML
    code_execution.py           subprocess execution; artifacts
  operations/
    registry.py                 @code registration for trusted functions
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
    planner_01.txt
    planner_02.txt                # present when planning is retried
  data/                           retrieved GeoJSON inputs
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
    solution.py
  results/
    llm_geo_result.json
    ...maps, charts, reports
```

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
- Never logged: secrets, raw prompts, generated code.

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
