"""Turn a RunReport into human-facing output: a Mermaid solution graph (as planned), a Mermaid
execution graph (as run, colored by outcome), and a Markdown report bundling both per test case.
"""
from __future__ import annotations

import json

import geopandas as gpd

from .models import DAGSpec, ExecutionResult, RunReport

_SHAPE = {"retrieval": ("([", "])"), "transformation": ("[", "]"), "synthesis": ("{{", "}}")}


def _label(*lines: str) -> str:
    return "<br/>".join(lines)


def _source(node, report: RunReport) -> str:
    if node.registry_id:
        return f"registry:{node.registry_id}"
    attempts = report.implementation_attempts.get(node.id)
    return f"llm-generated (x{attempts} attempt{'s' if attempts != 1 else ''})" if attempts else "llm-generated"


def mermaid_solution_graph(report: RunReport) -> str:
    """The planned DAG: node kind (shape) + registry vs. LLM-generated (color)."""
    lines = ["flowchart TD"]
    for node in report.dag.nodes:
        open_, close_ = _SHAPE[node.kind.value]
        label = _label(node.id, f"({node.kind.value})", _source(node, report))
        lines.append(f'    {node.id}{open_}"{label}"{close_}')
    for node in report.dag.nodes:
        for dep in node.depends_on:
            lines.append(f"    {dep} --> {node.id}")
    lines += [
        "    classDef registry fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;",
        "    classDef generated fill:#ede9fe,stroke:#7c3aed,color:#4c1d95;",
    ]
    for node in report.dag.nodes:
        lines.append(f"    class {node.id} {'registry' if node.registry_id else 'generated'};")
    return "\n".join(lines)


def mermaid_execution_graph(dag: DAGSpec, result: ExecutionResult) -> str:
    """The same DAG colored by what actually happened at run time: ok/error/skipped + timing."""
    lines = ["flowchart TD"]
    for node in dag.nodes:
        open_, close_ = _SHAPE[node.kind.value]
        status = result.node_status.get(node.id, "skipped")
        dur = result.node_duration_ms.get(node.id)
        label = _label(node.id, status.upper(), f"{dur:.0f}ms" if dur is not None else "-")
        lines.append(f'    {node.id}{open_}"{label}"{close_}')
    for node in dag.nodes:
        for dep in node.depends_on:
            lines.append(f"    {dep} --> {node.id}")
    lines += [
        "    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d;",
        "    classDef err fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;",
        "    classDef cached fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e;",
        "    classDef skipped fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray: 4 3;",
    ]
    for node in dag.nodes:
        cls = {"ok": "ok", "error": "err", "cached": "cached"}.get(result.node_status.get(node.id, "skipped"), "skipped")
        lines.append(f"    class {node.id} {cls};")
    return "\n".join(lines)


def agent_run_stats(report: RunReport) -> str:
    """How this run actually moved through the LangGraph control-flow graph below."""
    initial = sum(1 for n in report.dag.nodes if not n.registry_id)
    repairs = max(report.implement_calls - initial, 0)
    return "\n".join([
        f"- `plan`: 1 call",
        f"- `implement_one`: {report.implement_calls} call(s) "
        f"({initial} initial LLM-generated node(s) + {repairs} repair call(s))",
        f"- `assemble`: {report.repair_attempts} call(s) (assemble/execute round(s))",
    ])


def linear_order(report: RunReport) -> str:
    return " -> ".join(report.result.node_order or [n.id for n in report.dag.nodes])


def terminal_output(report: RunReport) -> tuple[str | None, dict]:
    """The final node's output -- the whole system's answer for this task."""
    order = report.result.node_order or list(report.result.outputs)
    if not order:
        return None, {}
    terminal = order[-1]
    return terminal, report.result.outputs.get(terminal, {})


def node_table(report: RunReport) -> str:
    result = report.result
    rows = ["| Node | Kind | Source | Status | Duration (ms) | Coder attempts |", "|---|---|---|---|---|---|"]
    for node in report.dag.nodes:
        dur = result.node_duration_ms.get(node.id)
        source = f"registry:{node.registry_id}" if node.registry_id else "llm-generated"
        rows.append(
            f"| `{node.id}` | {node.kind.value} | {source} | {result.node_status.get(node.id, 'skipped')} | "
            f"{f'{dur:.1f}' if dur is not None else '-'} | {report.implementation_attempts.get(node.id, '-')} |"
        )
    return "\n".join(rows)


def summarize_value(value, max_items: int = 6):
    """Collapse long lists (coordinate arrays, feature lists, ...) so the report stays scannable."""
    if isinstance(value, gpd.GeoDataFrame):
        return {"type": "GeoDataFrame", "rows": len(value), "columns": list(value.columns), "crs": str(value.crs)}
    if isinstance(value, dict):
        return {k: summarize_value(v, max_items) for k, v in value.items()}
    if isinstance(value, list):
        head = [summarize_value(v, max_items) for v in value[:max_items]]
        return head + [f"... {len(value) - max_items} more"] if len(value) > max_items else head
    return value


def _pretty(payload, limit: int = 600) -> str:
    text = json.dumps(summarize_value(payload), indent=2, default=str)
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def _pretty_outputs(report: RunReport, limit: int = 1500) -> str:
    return _pretty(terminal_output(report)[1], limit)


def node_io_section(report: RunReport) -> list[str]:
    """Per node, in execution order: the resolved inputs it received and the outputs it produced
    (summarized + truncated, GeoDataFrames collapsed to shape/columns/crs)."""
    result = report.result
    lines: list[str] = []
    for node_id in result.node_order or list(result.outputs):
        status = result.node_status.get(node_id, "?")
        dur = result.node_duration_ms.get(node_id)
        lines.append(f"#### `{node_id}` -- {status}{f' ({dur:.0f}ms)' if dur is not None else ''}")
        if status == "cached":
            lines.append("_Output reused from the previous execution round (not re-executed)._")
        elif node_id in result.node_inputs:
            lines += ["Inputs:", "```json", _pretty(result.node_inputs[node_id]), "```"]
        if node_id in result.outputs:
            lines += ["Outputs:", "```json", _pretty(result.outputs[node_id]), "```"]
        elif status == "error":
            lines.append(f"Outputs: none -- failed with `{' '.join((result.error or '').split())[:200]}`")
        lines.append("")
    return lines


def _sections(report: RunReport, include_agent_graph: bool = True) -> list[str]:
    """The shared body of a per-run report: task, output, graphs, node I/O, agent stats, node table."""
    lines = [
        "### Input task",
        f"> {report.task}",
        "",
        f"### Output ({terminal_output(report)[0] or '-'})",
        "```json",
        _pretty_outputs(report),
        "```",
        "",
        "### Solution graph (as planned)",
        "```mermaid",
        mermaid_solution_graph(report),
        "```",
        "",
        "### Execution graph (as run)",
        "```mermaid",
        mermaid_execution_graph(report.dag, report.result),
        "```",
        "",
        "### Node inputs/outputs (execution order)",
        *node_io_section(report),
    ]
    if include_agent_graph:
        lines += [
            "### Agent orchestration graph (LangGraph control flow)",
            "```mermaid",
            report.agent_graph_mermaid,
            "```",
            "This run's path through it:",
        ]
    else:
        lines += [
            "### Agent control flow",
            "This run's path through the shared orchestration graph (rendered once at the top of this report):",
        ]
    lines += [
        agent_run_stats(report),
        "",
        "### Node metadata",
        node_table(report),
        "",
    ]
    return lines


def case_section(name: str, report: RunReport, expect_success: bool, ok: bool, detail: str) -> str:
    lines = [
        f"## {name} - {'PASS' if ok else 'FAIL'}",
        "",
        f"- **Duration:** {report.duration_ms / 1000:.1f}s &nbsp; "
        f"**DAG-level repair rounds:** {report.repair_attempts} &nbsp; "
        f"**Nodes:** {len(report.dag.nodes)} &nbsp; "
        f"**Expected:** {'success' if expect_success else 'graceful failure'}",
        f"- **Outcome:** {detail}",
    ]
    if report.artifacts_dir:
        lines.append(f"- **Debug bundle:** `{report.artifacts_dir}`")
    return "\n".join(lines + ["", *_sections(report, include_agent_graph=False)])


def single_run_markdown(report: RunReport, generated_at: str) -> str:
    """A standalone report for one run, written into its debug bundle as report.md."""
    ok = report.result.success
    lines = [
        f"# Run report - {'SUCCESS' if ok else 'FAILED'}",
        "",
        f"Generated: {generated_at}",
        "",
        f"- **Duration:** {report.duration_ms / 1000:.1f}s &nbsp; "
        f"**DAG-level repair rounds:** {report.repair_attempts} &nbsp; "
        f"**Nodes:** {len(report.dag.nodes)}",
    ]
    if not ok:
        lines.append(f"- **Error:** {report.result.error}")
        if report.result.failing_node_ids:
            failing = ", ".join(f"`{n}`" for n in report.result.failing_node_ids)
            lines.append(f"- **Failing node(s):** {failing}")
    return "\n".join(lines + ["", *_sections(report)]) + "\n"


def full_report(run_at: str, cases: list[dict]) -> str:
    """`cases`: list of {"name", "report" (RunReport|None), "expect_success", "ok", "detail"}."""
    passed = sum(1 for c in cases if c["ok"])
    failed = len(cases) - passed
    lines = [
        "# LLM-Geo Agentic Workflow - Run Report",
        "",
        f"Generated: {run_at}",
        "",
        f"**Total:** {len(cases)} &nbsp; **Passed:** {passed} &nbsp; **Failed:** {failed} &nbsp; "
        f"**Status:** {'ALL PASSED' if failed == 0 else 'FAILURES PRESENT'}",
        "",
        "| # | Case | Status | Duration | DAG nodes | Repair rounds |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(cases, 1):
        r: RunReport | None = c["report"]
        dur = f"{r.duration_ms / 1000:.1f}s" if r else "-"
        nodes = str(len(r.dag.nodes)) if r else "-"
        repairs = str(r.repair_attempts) if r else "-"
        lines.append(f"| {i:02d} | {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {dur} | {nodes} | {repairs} |")
    lines.append("")

    # The LangGraph control flow is identical for every case -- render it exactly once up front.
    first = next((c["report"] for c in cases if c["report"] and c["report"].agent_graph_mermaid), None)
    if first is not None:
        lines += [
            "## Agent orchestration graph (LangGraph control flow)",
            "",
            "All cases share this control flow; each case section below lists its own path through it.",
            "",
            "```mermaid",
            first.agent_graph_mermaid,
            "```",
            "",
        ]

    for c in cases:
        if c["report"] is None:
            lines += [f"## {c['name']} - {'PASS' if c['ok'] else 'FAIL'}", "", f"- **Outcome:** {c['detail']}", ""]
        else:
            lines.append(case_section(c["name"], c["report"], c["expect_success"], c["ok"], c["detail"]))
    return "\n".join(lines)
