"""Per-run debug bundle: every prompt, code attempt, contract result, execution error (with full
traceback) and trace of one run, laid out under `output/<task_name>/<timestamp>/` so a human can
navigate straight from "it failed" to the exact code, prompt and traceback that caused it.

Layout of one bundle:

    output/<task_name>/<timestamp>/
    |-- README.md                    # start here: status, error index, navigation guide, file tree
    |-- run.json                     # machine-readable run summary
    |-- report.md                    # full human report: outputs, Mermaid graphs, node table
    |-- task.txt                     # the raw input task
    |-- trace.jsonl                  # step-by-step trace records (full tracebacks on errors)
    |-- trace.log                    # the console trace lines, mirrored to a file
    |-- plan/
    |   |-- dag.json                 # the planned DAGSpec (every node contract)
    |   |-- solution_graph.mmd       # planned DAG as Mermaid
    |   `-- prompts/                 # planner system + user prompt
    |-- nodes/<node_id>/             # one dir per LLM-implemented node
    |   |-- spec.json                # the node's contract (inputs/outputs/params)
    |   |-- system_prompt.md         # coder system prompt for this node
    |   |-- final.py                 # accepted (or last attempted) code
    |   `-- round_RR/                # RR = graph-level round (1 = initial, 2+ = repairs)
    |       |-- result.txt           # PASS/FAIL after how many attempts
    |       `-- attempt_AA/          # every coder attempt inside the round:
    |           |-- code.py          #   the generated code
    |           |-- prompt.md        #   the user prompt (incl. repair feedback)
    |           |-- contract.txt     #   PASS, or FAIL + full contract-test traceback
    |           |-- transcript.md    #   the agent's message/tool-call transcript
    |           |-- edits.json       #   repair rounds only: the find/replace edits returned
    |           `-- notes.md         #   the coder's own notes, if any
    |-- execution/attempt_AA/        # one dir per assemble/execute round
    |   |-- result.json              # ExecutionResult (status, order, timings)
    |   |-- inputs.json              # summarized resolved inputs per executed node
    |   |-- outputs.json             # summarized per-node outputs
    |   |-- execution_graph.mmd      # as-run DAG colored by outcome
    |   `-- traceback.txt            # full traceback of the failing node, if any
    `-- errors/
        |-- summary.md               # every error, chronological, linked to detail files
        |-- errors.jsonl             # the same, machine-readable
        `-- NNN_<stage>[_<node>].md  # one file per error: context + message + full traceback
"""
from __future__ import annotations

import datetime as dt
import json
import re
import threading
from pathlib import Path
from typing import Any

from .models import DAGSpec, ExecutionResult, NodeSpec, RunReport
from .report import mermaid_execution_graph, mermaid_solution_graph, single_run_markdown, summarize_value


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len].rstrip("_") or "task"


def transcript_markdown(messages: list[Any] | None) -> str:
    """Render an agent message transcript (LangChain BaseMessages, duck-typed) as Markdown."""
    lines: list[str] = []
    for message in messages or []:
        role = getattr(message, "type", None) or type(message).__name__
        lines.append(f"## {role}")
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, default=str)
        if content.strip():
            lines.append(content.strip())
        for call in getattr(message, "tool_calls", None) or []:
            args = json.dumps(call.get("args", {}), indent=2, default=str)
            lines.append(f"**tool call** `{call.get('name')}`:\n```json\n{args}\n```")
        lines.append("")
    return "\n".join(lines).strip() + "\n" if lines else ""


class RunArtifacts:
    """Collects everything produced during one run into a navigable on-disk bundle."""

    def __init__(self, task_name: str, root: str | Path = "output"):
        self.task_name = slugify(task_name)
        self.started_at = dt.datetime.now()
        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        base = Path(root) / self.task_name
        directory, suffix = base / stamp, 1
        while directory.exists():
            directory, suffix = base / f"{stamp}-{suffix}", suffix + 1
        directory.mkdir(parents=True)
        self.dir = directory
        self.task = ""
        self.errors: list[dict] = []
        self._lock = threading.Lock()
        self._node_rounds: dict[str, int] = {}
        self._node_results: dict[str, dict] = {}
        self._execution_attempts = 0

    # ------------------------------------------------------------------ low-level helpers
    def write(self, relpath: str, text: str) -> Path:
        path = self.dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, relpath: str, obj: Any) -> Path:
        return self.write(relpath, json.dumps(obj, indent=2, default=str) + "\n")

    # ------------------------------------------------------------------ error tracing
    def record_error(
        self,
        stage: str,
        message: str,
        *,
        node_id: str | None = None,
        round: int | None = None,
        attempt: int | None = None,
        traceback_text: str | None = None,
    ) -> dict:
        """Append one error to the chronological error log and write its full-detail file."""
        with self._lock:
            n = len(self.errors) + 1
            record = {
                "n": n,
                "at": dt.datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "node_id": node_id,
                "round": round,
                "attempt": attempt,
                "message": message,
                "detail_file": None,
            }
            name = f"{n:03d}_{slugify(stage, 24)}" + (f"_{slugify(node_id, 32)}" if node_id else "")
            record["detail_file"] = f"errors/{name}.md"
            self.errors.append(record)
            errors_jsonl = self.dir / "errors" / "errors.jsonl"
            errors_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with errors_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "traceback": traceback_text}, default=str) + "\n")

        where = " / ".join(
            part for part in (
                f"node `{node_id}`" if node_id else None,
                f"round {round}" if round else None,
                f"attempt {attempt}" if attempt else None,
            ) if part
        )
        detail = [
            f"# Error {n:03d} - {stage}" + (f" ({where})" if where else ""),
            "",
            f"- **When:** {record['at']}",
            f"- **Stage:** {stage}",
        ]
        if node_id:
            has_node_dir = (self.dir / "nodes" / node_id).is_dir()
            link = f" (see [`nodes/{node_id}/`](../nodes/{node_id}/))" if has_node_dir else ""
            detail.append(f"- **Node:** `{node_id}`{link}")
        if round:
            detail.append(f"- **Round:** {round}")
        if attempt:
            detail.append(f"- **Attempt:** {attempt}")
        detail += ["", "## Message", "", "```", message, "```"]
        if traceback_text:
            detail += ["", "## Full traceback", "", "```", traceback_text.rstrip(), "```"]
        self.write(record["detail_file"], "\n".join(detail) + "\n")
        return record

    # ------------------------------------------------------------------ phase hooks
    def save_task(self, task: str) -> None:
        self.task = task
        self.write("task.txt", task + "\n")

    def save_planner_prompts(self, system_prompt: str, user_prompt: str) -> None:
        self.write("plan/prompts/system.md", system_prompt)
        self.write("plan/prompts/user.md", user_prompt)

    def save_plan(self, dag: DAGSpec) -> None:
        self.write("plan/dag.json", dag.model_dump_json(indent=2) + "\n")

    def begin_node_round(self, node: NodeSpec, system_prompt: str) -> int:
        """Open a new coder round for a node (1 = initial implementation, 2+ = graph-level repairs)."""
        with self._lock:
            round = self._node_rounds.get(node.id, 0) + 1
            self._node_rounds[node.id] = round
        if round == 1:
            self.write(f"nodes/{node.id}/spec.json", node.model_dump_json(indent=2) + "\n")
            self.write(f"nodes/{node.id}/system_prompt.md", system_prompt)
        return round

    def save_coder_attempt(
        self,
        node_id: str,
        round: int,
        attempt: int,
        *,
        code: str,
        user_prompt: str,
        ok: bool,
        error: str | None = None,
        notes: str = "",
        transcript_md: str = "",
        edits_json: str = "",
    ) -> None:
        base = f"nodes/{node_id}/round_{round:02d}/attempt_{attempt:02d}"
        self.write(f"{base}/code.py", code)
        self.write(f"{base}/prompt.md", user_prompt)
        self.write(f"{base}/contract.txt", "PASS\n" if ok else f"FAIL\n\n{error or ''}".rstrip() + "\n")
        if notes:
            self.write(f"{base}/notes.md", notes)
        if transcript_md:
            self.write(f"{base}/transcript.md", transcript_md)
        if edits_json:
            self.write(f"{base}/edits.json", edits_json)
        if not ok:
            self.record_error(
                "contract_test",
                f"node '{node_id}' failed its contract test (see {base}/code.py)",
                node_id=node_id, round=round, attempt=attempt, traceback_text=error,
            )

    def save_node_result(self, node_id: str, round: int, code: str, ok: bool, attempts: int) -> None:
        verdict = f"{'PASS' if ok else 'FAIL'} after {attempts} attempt(s)\n"
        self.write(f"nodes/{node_id}/round_{round:02d}/result.txt", verdict)
        self.write(f"nodes/{node_id}/final.py", code)
        with self._lock:
            self._node_results[node_id] = {"round": round, "attempts": attempts, "contract_ok": ok}

    def save_execution(self, attempt: int, result: ExecutionResult, dag: DAGSpec | None = None) -> None:
        with self._lock:
            self._execution_attempts = max(self._execution_attempts, attempt)
        base = f"execution/attempt_{attempt:02d}"
        self.write_json(f"{base}/result.json", result.model_dump(exclude={"outputs", "node_inputs"}))
        self.write_json(f"{base}/outputs.json", {k: summarize_value(v) for k, v in result.outputs.items()})
        self.write_json(f"{base}/inputs.json", {k: summarize_value(v) for k, v in result.node_inputs.items()})
        if dag is not None:
            self.write(f"{base}/execution_graph.mmd", mermaid_execution_graph(dag, result) + "\n")
        if result.error_traceback:
            self.write(f"{base}/traceback.txt", result.error_traceback)
        if not result.success and not result.error_traceback:
            # Non-exception failures (cycle, missing terminal outputs) are not seen by the tracer's
            # error hook, so record them here to keep the error log complete.
            self.record_error(
                "execute", result.error or "execution failed",
                node_id=(result.failing_node_ids or [None])[0], attempt=attempt,
            )

    # ------------------------------------------------------------------ finalization
    def finalize(self, report: RunReport | None) -> Path:
        """Write the derived views (report, error summary, run.json, README index). Safe to call
        with report=None when the run crashed before producing one."""
        finished_at = dt.datetime.now()
        generated = finished_at.isoformat(timespec="seconds")
        if report is not None:
            self.write("plan/solution_graph.mmd", mermaid_solution_graph(report) + "\n")
            self.write("report.md", single_run_markdown(report, generated))
        self._write_error_summary()
        self._write_run_json(report, finished_at)
        self._write_readme(report, finished_at)
        return self.dir

    def _write_error_summary(self) -> None:
        if not self.errors:
            self.write("errors/summary.md", "# Error summary\n\nNo errors recorded during this run.\n")
            return
        lines = [
            "# Error summary",
            "",
            f"{len(self.errors)} error(s), chronological -- the first error is usually closest to the "
            "root cause; the last one is what finally failed (if the run failed).",
            "",
            "| # | Stage | Node | Round | Attempt | Message | Detail |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in self.errors:
            message = " ".join((e["message"] or "").split())
            message = message if len(message) <= 100 else message[:97] + "..."
            node = "`{}`".format(e["node_id"]) if e["node_id"] else "-"
            detail_name = Path(e["detail_file"]).name
            lines.append(
                f"| {e['n']:03d} | {e['stage']} | {node} "
                f"| {e['round'] or '-'} | {e['attempt'] or '-'} | {message} "
                f"| [{detail_name}]({detail_name}) |"
            )
        self.write("errors/summary.md", "\n".join(lines) + "\n")

    def _write_run_json(self, report: RunReport | None, finished_at: dt.datetime) -> None:
        success = bool(report and report.result.success)
        self.write_json("run.json", {
            "task_name": self.task_name,
            "task": self.task,
            "bundle_dir": str(self.dir),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_s": round((finished_at - self.started_at).total_seconds(), 1),
            "success": success,
            "error": (report.result.error if report else "run crashed before producing a report"),
            "failing_node_ids": report.result.failing_node_ids if report else [],
            "repair_rounds": report.repair_attempts if report else None,
            "execution_attempts": self._execution_attempts,
            "nodes": self._node_results,
            "error_count": len(self.errors),
            "errors": self.errors,
        })

    def _write_readme(self, report: RunReport | None, finished_at: dt.datetime) -> None:
        success = bool(report and report.result.success)
        status = "SUCCESS" if success else "FAILED"
        error = None if success else (report.result.error if report else "run crashed before producing a report")
        duration = (finished_at - self.started_at).total_seconds()

        lines = [
            f"# Debug bundle: {self.task_name} - {status}",
            "",
            "| | |",
            "|---|---|",
            f"| Task | {' '.join(self.task.split())} |",
            f"| Started | {self.started_at.isoformat(timespec='seconds')} |",
            f"| Duration | {duration:.1f}s |",
            f"| Status | {status} |",
        ]
        if error:
            lines.append(f"| Error | {' '.join(str(error).split())[:200]} |")
        if report:
            failing = ", ".join(f"`{n}`" for n in report.result.failing_node_ids)
            lines.append(f"| Repair rounds | {report.repair_attempts} |")
            if failing:
                lines.append(f"| Failing node(s) | {failing} |")
        lines.append(f"| Errors recorded | {len(self.errors)} (see [errors/summary.md](errors/summary.md)) |")

        lines += ["", "## Where to look first", ""]
        if success:
            lines += [
                "1. [report.md](report.md) - final outputs, planned vs. executed graph, per-node table.",
                "2. [errors/summary.md](errors/summary.md) - any bumps along the way "
                "(contract-test failures the coder recovered from, repaired nodes).",
                "3. [trace.log](trace.log) - the full step-by-step timeline with durations.",
            ]
        else:
            failing_node = (report.result.failing_node_ids[0] if report and report.result.failing_node_ids else None)
            if failing_node and not (self.dir / "nodes" / failing_node).is_dir():
                failing_node = None  # registry-implemented node: there is no coder dir to link
            last_exec = f"execution/attempt_{self._execution_attempts:02d}/traceback.txt"
            traceback_line = (
                f"2. [{last_exec}]({last_exec}) - full traceback of the failing node run."
                if self._execution_attempts and (self.dir / last_exec).exists()
                else "2. `execution/attempt_XX/traceback.txt` - full traceback of the failing node run."
            )
            lines += [
                "1. [errors/summary.md](errors/summary.md) - all errors in order; the **last** one is what "
                "finally failed, earlier ones show how the coder/repair loop struggled.",
                traceback_line,
                (
                    f"3. [`nodes/{failing_node}/`](nodes/{failing_node}/) - compare `final.py` against "
                    f"`spec.json`, and each `round_*/attempt_*/contract.txt` against its `code.py`."
                    if failing_node else
                    "3. `nodes/<node_id>/` - compare `final.py` against `spec.json`, and each "
                    "`round_*/attempt_*/contract.txt` against its `code.py`."
                ),
                "4. [trace.log](trace.log) - the timeline: which phase/node broke, and when.",
            ]

        lines += [
            "",
            "## Directory guide",
            "",
            "| Path | Contents |",
            "|---|---|",
            "| `report.md` | Full human report: outputs, Mermaid solution/execution graphs, node table |",
            "| `run.json` | Machine-readable run summary (status, timings, per-node attempts, error index) |",
            "| `task.txt` | The raw input task |",
            "| `errors/summary.md` | Every error chronologically, linked to full-detail files |",
            "| `errors/NNN_*.md` | One file per error: context + message + full traceback |",
            "| `errors/errors.jsonl` | The same errors, machine-readable |",
            "| `plan/dag.json` | The planned DAG: every node's contract (inputs/outputs/params/deps) |",
            "| `plan/solution_graph.mmd` | Planned DAG as Mermaid |",
            "| `plan/prompts/` | Planner system + user prompt |",
            "| `nodes/<id>/spec.json` | That node's contract |",
            "| `nodes/<id>/system_prompt.md` | Coder system prompt for the node |",
            "| `nodes/<id>/final.py` | Accepted (or last attempted) implementation |",
            "| `nodes/<id>/round_RR/attempt_AA/` | Per coder attempt: `code.py`, `prompt.md`, `contract.txt`, `transcript.md` |",
            "| `execution/attempt_AA/` | Per execute round: `result.json`, `inputs.json`, `outputs.json`, `execution_graph.mmd`, `traceback.txt` |",
            "| `trace.jsonl` / `trace.log` | Step-by-step trace with durations (JSONL + console mirror) |",
            "",
            "## Bundle contents",
            "",
            "```",
            _tree(self.dir),
            "```",
        ]
        self.write("README.md", "\n".join(lines) + "\n")


def _tree(root: Path) -> str:
    lines = [root.name + "/"]

    def walk(directory: Path, prefix: str) -> None:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if last else "|   "))

    walk(root, "")
    return "\n".join(lines)
