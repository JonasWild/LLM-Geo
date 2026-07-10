"""Shared GIS instructions and run-scoped prompt persistence."""

from __future__ import annotations

import re
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

GIS_RULES = """
- Never invent data columns, coordinate reference systems, paths, or results.
- Use only the system-retrieved local GeoJSON source paths supplied in the workflow.
- Never issue a new HTTP request or query an external data API from generated code.
- Treat retrieved GeoJSON files as read-only inputs. Never write maps, charts,
	reports, manifests, or derived data beside an input source.
- The program runs with the run's results directory as its working directory. Write
	every generated artifact using a relative path in that directory.
- Inspect data before planning and preserve observed field names.
- Reproject spatial layers to compatible projected CRSs before distance/area work.
- Preserve identifier semantics, especially leading zeros in FIPS/GEOID fields.
- Handle nulls, invalid geometries, join cardinality, and duplicate spatial joins.
- Every operation must have explicit inputs, outputs, and validation criteria.
- Maps and charts must include meaningful titles, units, legends/colorbars, and be saved.
- Use current Pandas, GeoPandas, Shapely, Rasterio, Matplotlib, SciPy, and Statsmodels APIs.
""".strip()


def save_prompt(
    save_dir: str | Path,
    *,
    stage: str,
    agent: str,
    subject: str,
    prompt: str,
) -> Path:
    """Save an LLM prompt with compact origin and attempt metadata."""
    prompt_directory = Path(save_dir) / "prompts"
    prompt_directory.mkdir(parents=True, exist_ok=True)
    stage_slug = _prompt_slug(stage)
    agent_slug = _prompt_slug(agent)
    subject_slug = _prompt_slug(subject, max_length=48)
    action, filename_subject = _prompt_filename_parts(
        stage_slug, agent_slug, subject_slug
    )
    stem = action if filename_subject is None else f"{action}_{filename_subject}"
    existing = list(prompt_directory.glob(f"*_{stem}_*.txt"))
    attempts = []
    for path in existing:
        match = re.search(r"_(\d+)\.txt$", path.name)
        if match:
            attempts.append(int(match.group(1)))
    attempt = max(attempts, default=0) + 1
    sequences = []
    for path in prompt_directory.glob("*.txt"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            sequences.append(int(match.group(1)))
    sequence = max(sequences, default=0) + 1
    path = prompt_directory / f"{sequence:03d}_{stem}_{attempt:02d}.txt"
    header = (
        f"Stage: {stage}\n"
        f"Agent: {agent}\n"
        f"Subject: {subject}\n"
        f"Attempt: {attempt}\n"
        f"Created: {datetime.now(timezone.utc).isoformat()}\n\n"
        "--- PROMPT ---\n\n"
    )
    path.write_text(header + prompt, encoding="utf-8")
    return path


def _prompt_slug(value: str, max_length: int = 32) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    slug = slug or "prompt"
    if len(slug) <= max_length:
        return slug
    digest = sha256(slug.encode("utf-8")).hexdigest()[:8]
    return f"{slug[: max_length - 9].rstrip('_')}_{digest}"


def _prompt_filename_parts(
    stage: str, agent: str, subject: str
) -> tuple[str, str | None]:
    if agent == "retriever":
        return "retrieve", None
    if agent == "planner":
        return "plan", None
    if stage == "ops" and agent == "coder":
        return "code", subject
    if stage == "ops" and agent == "reviewer":
        return "review", subject
    if agent == "assembler":
        return "assemble", None
    if stage == "assemble" and agent == "reviewer":
        return "review", "assemble"
    if stage == "direct" and agent == "coder":
        return "direct", None
    if stage == "direct" and agent == "reviewer":
        return "review", "direct"
    if agent == "debugger":
        return "debug", None
    if agent == "validator":
        return "validate", None
    return agent, subject
