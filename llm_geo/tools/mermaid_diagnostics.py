"""Mermaid visualizations for the complete system and an observed execution."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langgraph.graph.state import CompiledStateGraph

from llm_geo.middleware.logging import get_logger


MERMAID_CLI_PACKAGE = "@mermaid-js/mermaid-cli@11.12.0"
FAILED_STATUSES = {
    "execution_failed",
    "failed",
    "plan_invalid",
    "retrieval_failed",
    "validation_failed",
}


def execution_event(
    trace: Sequence[dict[str, Any]],
    node: str,
    status: str,
    *,
    started_at: str | None = None,
    duration_seconds: float = 0.0,
    exception_type: str | None = None,
) -> dict[str, Any]:
    """Create a compact, non-sensitive event for one top-level graph node."""
    occurrence = sum(event.get("node") == node for event in trace) + 1
    if exception_type:
        outcome = "exception"
    elif status in FAILED_STATUSES or status.endswith("_failed"):
        outcome = "failure"
    else:
        outcome = "success"
    return {
        "sequence": len(trace) + 1,
        "node": node,
        "occurrence": occurrence,
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "status": status,
        "outcome": outcome,
        "exception_type": exception_type,
    }


def write_system_graph_artifacts(
    graph: CompiledStateGraph,
    save_dir: Path,
) -> list[str]:
    """Persist Mermaid source and PNG for every possible top-level route."""
    workflow_directory = save_dir / "workflow"
    workflow_directory.mkdir(parents=True, exist_ok=True)
    source_path = workflow_directory / "system.mmd"
    png_path = workflow_directory / "system.png"
    source_path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    artifacts = [str(source_path)]
    if render_mermaid_png(source_path, png_path):
        artifacts.append(str(png_path))
    return artifacts


def write_execution_graph_artifacts(
    trace: Sequence[dict[str, Any]],
    save_dir: Path,
) -> list[str]:
    """Persist Mermaid source and PNG for the chronological route actually taken."""
    workflow_directory = save_dir / "workflow"
    workflow_directory.mkdir(parents=True, exist_ok=True)
    source_path = workflow_directory / "execution.mmd"
    png_path = workflow_directory / "execution.png"
    source_path.write_text(execution_mermaid(trace), encoding="utf-8")
    artifacts = [str(source_path)]
    if render_mermaid_png(source_path, png_path):
        artifacts.append(str(png_path))
    return artifacts


def execution_mermaid(trace: Sequence[dict[str, Any]]) -> str:
    """Build a Mermaid flowchart whose nodes are execution occurrences."""
    lines = [
        "flowchart TD",
        '    start(["START"]):::boundary',
    ]
    previous = "start"
    for index, event in enumerate(trace, start=1):
        node_id = f"event_{index}"
        node = _escape_label(str(event.get("node", "unknown")))
        occurrence = int(event.get("occurrence", 1))
        status = _escape_label(str(event.get("status", "unknown")))
        duration = float(event.get("duration_seconds", 0.0))
        label = f"{node} #{occurrence}<br/>{status}<br/>{duration:.3f}s"
        outcome = str(event.get("outcome", "success"))
        style = outcome if outcome in {"success", "failure", "exception"} else "success"
        lines.append(f'    {node_id}["{label}"]:::{style}')
        lines.append(f"    {previous} --> {node_id}")
        previous = node_id
    lines.append('    finish(["END"]):::boundary')
    lines.append(f"    {previous} --> finish")
    lines.extend(
        [
            "    classDef boundary fill:#e5e7eb,stroke:#4b5563,color:#111827",
            "    classDef success fill:#dcfce7,stroke:#16a34a,color:#14532d",
            "    classDef failure fill:#fef3c7,stroke:#d97706,color:#78350f",
            "    classDef exception fill:#fee2e2,stroke:#dc2626,color:#7f1d1d",
        ]
    )
    return "\n".join(lines) + "\n"


def render_mermaid_png(source_path: Path, png_path: Path) -> bool:
    """Render through a local Mermaid CLI, without making graph production fatal."""
    command = _mermaid_command()
    if command is None:
        get_logger().warning(
            "Mermaid PNG skipped | install Node.js/npm or mmdc | source=%s",
            source_path,
        )
        return False
    try:
        process = subprocess.run(
            [
                *command,
                "-i",
                str(source_path),
                "-o",
                str(png_path),
                "-b",
                "white",
                "-s",
                "2",
            ],
            cwd=source_path.parent,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        get_logger().warning(
            "Mermaid PNG rendering failed | reason=%s | source=%s",
            type(error).__name__,
            source_path,
        )
        return False
    if process.returncode != 0 or not png_path.is_file():
        detail = (process.stderr or process.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"return code {process.returncode}"
        get_logger().warning(
            "Mermaid PNG rendering failed | reason=%s | source=%s",
            reason[:300],
            source_path,
        )
        return False
    get_logger().info("Mermaid graph saved | path=%s", png_path)
    return True


def _mermaid_command() -> list[str] | None:
    configured = os.environ.get("LLM_GEO_MERMAID_CLI", "").strip()
    if configured:
        return shlex.split(configured, posix=os.name != "nt")
    executable = shutil.which("mmdc")
    if executable:
        return [executable]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", MERMAID_CLI_PACKAGE]
    return None


def _escape_label(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )
