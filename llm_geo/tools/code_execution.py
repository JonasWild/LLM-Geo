"""Generated-program execution and artifact utilities."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llm_geo.middleware.logging import get_logger


def snapshot_files(directory: Path) -> set[str]:
    return {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }


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
    program_path = code_directory / "solution.py"
    program_path.write_text(code, encoding="utf-8")
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
