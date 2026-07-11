"""Generated-program execution and artifact utilities."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_geo.middleware.logging import get_logger


def snapshot_files(directory: Path) -> set[str]:
    return {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }


def _atomic_write(path: Path, content: str) -> None:
    """Replace a text file atomically after writing it beside the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _revision_number(path: Path) -> int:
    match = re.match(r"^(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def save_code_revision(
    save_dir: Path,
    code: str,
    source: str,
    *,
    parent_revision: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one immutable full-program revision and update the revision index."""
    revision_directory = save_dir / "code" / "revisions"
    revision_directory.mkdir(parents=True, exist_ok=True)
    revision = max(
        (_revision_number(path) for path in revision_directory.glob("*.py")),
        default=0,
    )
    safe_source = re.sub(r"[^A-Za-z0-9_-]+", "_", source).strip("_") or "unknown"
    while True:
        revision += 1
        path = revision_directory / f"{revision:03d}_{safe_source}.py"
        try:
            with path.open("x", encoding="utf-8") as revision_file:
                revision_file.write(code)
                revision_file.flush()
                os.fsync(revision_file.fileno())
            break
        except FileExistsError:
            continue
    record: dict[str, Any] = {
        "revision": revision,
        "source": source,
        "path": path.relative_to(save_dir).as_posix(),
        "parent_revision": parent_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        record["metadata"] = metadata
    index_path = save_dir / "code" / "revisions.jsonl"
    with index_path.open("a", encoding="utf-8") as index:
        index.write(json.dumps(record, ensure_ascii=False) + "\n")
        index.flush()
        os.fsync(index.fileno())
    return record


def publish_solution(save_dir: Path, code: str) -> Path:
    """Atomically publish the current candidate at the stable solution path."""
    path = save_dir / "code" / "solution.py"
    _atomic_write(path, code)
    return path


def save_execution_attempt(save_dir: Path, attempt: int, code: str) -> Path:
    """Persist the exact source submitted for an execution attempt."""
    path = save_dir / "code" / "executions" / f"attempt_{attempt:03d}.py"
    if path.exists():
        if path.read_text(encoding="utf-8") != code:
            raise RuntimeError(f"Execution attempt {attempt} already has different code")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as attempt_file:
            attempt_file.write(code)
            attempt_file.flush()
            os.fsync(attempt_file.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != code:
            raise RuntimeError(f"Execution attempt {attempt} already has different code")
    return path


def save_execution_result(
    save_dir: Path,
    attempt: int,
    revision: int | None,
    execution: dict[str, Any],
    *,
    executed: bool = True,
) -> dict[str, Any]:
    """Persist the complete outcome metadata for one execution attempt."""
    source_path = save_dir / "code" / "executions" / f"attempt_{attempt:03d}.py"
    result_path = source_path.with_suffix(".json")
    record = {
        **execution,
        "attempt": attempt,
        "revision": revision,
        "executed": executed,
        "source_path": source_path.relative_to(save_dir).as_posix(),
        "result_path": result_path.relative_to(save_dir).as_posix(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(result_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


def execute_code(
    code: str,
    save_dir: Path,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Execute generated code in a separate process and capture evidence."""
    code_directory = save_dir / "code"
    results_directory = save_dir / "results"
    code_directory.mkdir(parents=True, exist_ok=True)
    results_directory.mkdir(parents=True, exist_ok=True)
    program_path = publish_solution(save_dir, code)
    before = snapshot_files(results_directory)
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(project_root)
        if not existing_path
        else str(project_root) + os.pathsep + existing_path
    )
    try:
        process = subprocess.run(
            [sys.executable, str(program_path)],
            cwd=results_directory,
            capture_output=True,
            env=environment,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.perf_counter() - started
        after = snapshot_files(results_directory)
        result = {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "new_files": sorted(after - before),
            "program_path": str(program_path),
            "duration_seconds": round(duration, 2),
        }
        if result["success"]:
            get_logger().info(
                "Program finished | duration=%.2fs | new_artifacts=%d",
                duration,
                len(result["new_files"]),
            )
            output = process.stdout.strip()
            if output:
                get_logger().info("Program result | %s", output[-500:].replace("\n", " | "))
        else:
            error_line = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "unknown error"
            get_logger().warning(
                "Program failed | returncode=%s | duration=%.2fs | error=%s",
                process.returncode,
                duration,
                error_line,
            )
        return result
    except subprocess.TimeoutExpired as error:
        duration = time.perf_counter() - started
        get_logger().warning(
            "Program timed out | limit=%ds | duration=%.2fs", timeout_seconds, duration
        )
        return {
            "success": False,
            "returncode": None,
            "stdout": error.stdout or "",
            "stderr": f"Execution timed out after {timeout_seconds} seconds.",
            "new_files": [],
            "program_path": str(program_path),
            "duration_seconds": round(duration, 2),
        }
