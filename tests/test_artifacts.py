"""Offline tests for the per-run debug bundle: no LLM calls involved."""
import json

import pytest

from llm_geo.artifacts import RunArtifacts, slugify, transcript_markdown
from llm_geo.models import DAGSpec, ExecutionResult, NodeKind, NodeSpec, RunReport
from llm_geo.trace import Tracer


def make_dag() -> DAGSpec:
    return DAGSpec(task="demo", nodes=[
        NodeSpec(id="load", kind=NodeKind.retrieval, description="load", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="summarize", kind=NodeKind.synthesis, description="summarize", depends_on=["load"],
                 inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"}),
    ])


def make_report(artifacts_dir: str = "") -> RunReport:
    result = ExecutionResult(
        success=False, outputs={"load": {"features": {"n": 1}}}, failing_node_ids=["summarize"],
        error="boom", error_traceback="Traceback (most recent call last):\n  ...\nValueError: boom",
        node_order=["load", "summarize"], node_status={"load": "ok", "summarize": "error"},
        node_duration_ms={"load": 1.0, "summarize": 2.0},
    )
    return RunReport(
        task="demo task", dag=make_dag(), repair_attempts=1, result=result, duration_ms=1234.5,
        artifacts_dir=artifacts_dir,
    )


def test_slugify():
    assert slugify("Buffer & summarize POINTS!") == "buffer_summarize_points"
    assert slugify("   ") == "task"
    assert len(slugify("x" * 200)) <= 60


def test_bundle_lifecycle(tmp_path):
    artifacts = RunArtifacts("My Demo Case", root=tmp_path)
    assert artifacts.dir.parent == tmp_path / "my_demo_case"

    artifacts.save_task("demo task")
    artifacts.save_planner_prompts("planner system", "demo task")
    dag = make_dag()
    artifacts.save_plan(dag)

    node = dag.nodes[1]
    round_no = artifacts.begin_node_round(node, "coder system prompt")
    assert round_no == 1
    artifacts.save_coder_attempt(
        node.id, round_no, 1, code="def run(**i): ...", user_prompt="Implement it.",
        ok=False, error="Traceback ...\nTypeError: nope", transcript_md="## ai\nhi\n",
    )
    artifacts.save_coder_attempt(
        node.id, round_no, 2, code="def run(**i): return {}", user_prompt="Fix it.", ok=True,
    )
    artifacts.save_node_result(node.id, round_no, "def run(**i): return {}", True, 2)
    assert artifacts.begin_node_round(node, "coder system prompt") == 2  # a repair round

    result = make_report().result
    artifacts.save_execution(1, result, dag)
    artifacts.record_error("exec", "ValueError: boom", node_id="summarize", traceback_text="Traceback ...")

    report = make_report(artifacts_dir=str(artifacts.dir))
    artifacts.finalize(report)

    d = artifacts.dir
    for rel in [
        "README.md", "run.json", "report.md", "task.txt", "errors/summary.md", "errors/errors.jsonl",
        "plan/dag.json", "plan/solution_graph.mmd", "plan/prompts/system.md", "plan/prompts/user.md",
        "nodes/summarize/spec.json", "nodes/summarize/system_prompt.md", "nodes/summarize/final.py",
        "nodes/summarize/round_01/result.txt",
        "nodes/summarize/round_01/attempt_01/code.py", "nodes/summarize/round_01/attempt_01/contract.txt",
        "nodes/summarize/round_01/attempt_01/prompt.md", "nodes/summarize/round_01/attempt_01/transcript.md",
        "nodes/summarize/round_01/attempt_02/contract.txt",
        "execution/attempt_01/result.json", "execution/attempt_01/outputs.json",
        "execution/attempt_01/execution_graph.mmd", "execution/attempt_01/traceback.txt",
    ]:
        assert (d / rel).exists(), f"missing {rel}"

    run_meta = json.loads((d / "run.json").read_text())
    assert run_meta["success"] is False
    assert run_meta["error_count"] == 2  # contract failure + exec error
    assert run_meta["nodes"]["summarize"]["attempts"] == 2

    # the failed attempt produced a numbered error detail file with the full traceback
    error_files = sorted((d / "errors").glob("0*_*.md"))
    assert len(error_files) == 2
    assert "TypeError: nope" in error_files[0].read_text()
    summary = (d / "errors" / "summary.md").read_text()
    assert "contract_test" in summary and "exec" in summary
    assert (d / "nodes/summarize/round_01/attempt_02/contract.txt").read_text() == "PASS\n"
    assert "FAILED" in (d / "README.md").read_text()


def test_finalize_without_report_still_writes_bundle(tmp_path):
    artifacts = RunArtifacts("crashed", root=tmp_path)
    artifacts.save_task("t")
    artifacts.record_error("plan", "RuntimeError: kaputt", traceback_text="Traceback ...")
    artifacts.finalize(None)
    assert (artifacts.dir / "README.md").exists()
    assert json.loads((artifacts.dir / "run.json").read_text())["success"] is False
    assert "kaputt" in (artifacts.dir / "errors" / "summary.md").read_text()


def test_tracer_on_error_hook_and_log_file(tmp_path):
    errors = []
    tracer = Tracer(
        path=tmp_path / "trace.jsonl", log_file=tmp_path / "trace.log",
        on_error=lambda phase, node_id, message, tb: errors.append((phase, node_id, message, tb)),
    )
    with tracer.span("exec", "n1"):
        pass
    with pytest.raises(ValueError):
        with tracer.span("exec", "n2"):
            raise ValueError("kapow")

    assert len(errors) == 1
    phase, node_id, message, tb = errors[0]
    assert (phase, node_id) == ("exec", "n2")
    assert "kapow" in message and "Traceback" in tb

    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert records[0]["status"] == "OK" and "traceback" not in records[0]
    assert records[1]["status"] == "ERR" and "kapow" in records[1]["traceback"]
    assert "n2" in (tmp_path / "trace.log").read_text()


def test_transcript_markdown_tolerates_plain_objects():
    class Msg:
        type = "ai"
        content = "hello"
        tool_calls = [{"name": "contract_test", "args": {"code": "def run(): ..."}, "id": "1"}]

    md = transcript_markdown([Msg()])
    assert "## ai" in md and "contract_test" in md and "def run" in md
    assert transcript_markdown([]) == ""
