"""Offline tests for report rendering and plan validation: no LLM calls involved."""
from llm_geo import planner, registry
from llm_geo.models import DAGSpec, ExecutionResult, NodeKind, NodeSpec, RunReport
from llm_geo.planner import registry_errors, single_graph_errors, wiring_errors
from llm_geo.report import full_report, single_run_markdown


def make_dag(extra_nodes: list[NodeSpec] | None = None) -> DAGSpec:
    return DAGSpec(task="demo", nodes=[
        NodeSpec(id="load", kind=NodeKind.retrieval, description="load", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="summarize", kind=NodeKind.synthesis, description="summarize", depends_on=["load"],
                 inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"}),
        *(extra_nodes or []),
    ])


def make_report() -> RunReport:
    result = ExecutionResult(
        success=True,
        outputs={"load": {"features": {"n": 3}}, "summarize": {"report": {"count": 3}}},
        node_order=["load", "summarize"],
        node_status={"load": "cached", "summarize": "ok"},
        node_duration_ms={"load": 0.0, "summarize": 2.5},
        node_inputs={"summarize": {"features": {"n": 3}}},
    )
    return RunReport(task="demo task", dag=make_dag(), repair_attempts=1, result=result,
                     duration_ms=1000.0, agent_graph_mermaid="graph TD;\n  plan --> assemble")


def test_report_shows_per_node_inputs_and_outputs():
    md = single_run_markdown(make_report(), "2026-01-01T00:00:00")
    assert "### Node inputs/outputs (execution order)" in md
    assert "#### `summarize` -- ok (3ms)" in md or "#### `summarize` -- ok (2ms)" in md
    assert '"features"' in md and '"count": 3' in md  # inputs and outputs rendered
    assert "reused from the previous execution round" in md  # cached node annotated
    # single-run report still embeds the control-flow graph exactly once
    assert md.count("plan --> assemble") == 1


def test_full_report_renders_agent_graph_once():
    cases = [
        {"name": "one", "report": make_report(), "expect_success": True, "ok": True, "detail": "ok"},
        {"name": "two", "report": make_report(), "expect_success": True, "ok": True, "detail": "ok"},
        {"name": "crashed", "report": None, "expect_success": True, "ok": False, "detail": "raised"},
    ]
    md = full_report("2026-01-01T00:00:00", cases)
    assert md.count("plan --> assemble") == 1  # shared graph rendered exactly once
    assert md.count("## Agent orchestration graph") == 1
    assert md.count("### Agent control flow") == 2  # per-case stats remain
    assert md.count("### Node inputs/outputs") == 2


def test_single_graph_errors():
    assert single_graph_errors(make_dag()) == []
    assert single_graph_errors(DAGSpec(task="t", nodes=[])) == ["the DAG has no nodes"]

    island = NodeSpec(id="stray", kind=NodeKind.synthesis, description="disconnected", outputs={"report": "dict"})
    errors = single_graph_errors(make_dag([island]))
    assert len(errors) == 1 and "2 disconnected graphs" in errors[0] and "stray" in errors[0]

    bad_dep = NodeSpec(id="x", kind=NodeKind.synthesis, description="x", depends_on=["nope"],
                       outputs={"report": "dict"})
    assert any("unknown node 'nope'" in e for e in single_graph_errors(make_dag([bad_dep])))

    dupe = make_dag()
    dupe.nodes.append(dupe.nodes[0].model_copy())
    assert any("duplicate node id 'load'" in e for e in single_graph_errors(dupe))


def test_wiring_errors():
    assert wiring_errors(make_dag()) == []

    # Same-named edge with incompatible coarse types.
    mismatch = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.retrieval, description="a", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"features": "dict"}, outputs={"report": "dict"}),
    ])
    errors = wiring_errors(mismatch)
    assert len(errors) == 1 and "expects dict but 'a' outputs GeoDataFrame" in errors[0]

    # int output feeding a float input is fine.
    widening = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.transformation, description="a", outputs={"count": "int"}),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"count": "float"}, outputs={"report": "dict"}),
    ])
    assert wiring_errors(widening) == []

    # Input fed by nothing: no dependency output, no param.
    unfed = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.retrieval, description="a", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"features": "GeoDataFrame", "threshold": "float"}, outputs={"report": "dict"}),
    ])
    errors = wiring_errors(unfed)
    assert len(errors) == 1 and "'threshold' is fed by no dependency output and no param" in errors[0]

    # Param literal of the wrong coarse type.
    bad_param = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.retrieval, description="a", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"features": "GeoDataFrame", "threshold": "float"},
                 outputs={"report": "dict"}, params={"threshold": "high"}),
    ])
    errors = wiring_errors(bad_param)
    assert len(errors) == 1 and "param literal is str" in errors[0]


def test_registry_errors(monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "fake_op", {
        "kind": "retrieval", "description": "fake",
        "inputs": {"query": "str"}, "outputs": {"features": "GeoDataFrame"},
        "fn": lambda **kwargs: {"features": None},
    })
    ok = DAGSpec(task="t", nodes=[
        NodeSpec(id="fetch", kind=NodeKind.retrieval, description="f", registry_id="fake_op",
                 inputs={"query": "str"}, outputs={"features": "GeoDataFrame"},
                 params={"query": "berlin"}),
    ])
    assert registry_errors(ok) == []

    unknown = ok.model_copy(deep=True)
    unknown.nodes[0].registry_id = "nope"
    assert any("unknown registry_id 'nope'" in e for e in registry_errors(unknown))

    drifted = DAGSpec(task="t", nodes=[
        NodeSpec(id="fetch", kind=NodeKind.retrieval, description="f", registry_id="fake_op",
                 inputs={"place": "str", "query": "int"}, outputs={"stuff": "dict"}),
    ])
    errors = registry_errors(drifted)
    assert any("has no input 'place'" in e for e in errors)
    assert any("registry input 'query' has type str, not int" in e for e in errors)
    assert any("outputs 'features', declare it" in e for e in errors)
    assert any("has no output 'stuff'" in e for e in errors)


def test_plan_retries_once_on_disconnected_dag(monkeypatch):
    disconnected = make_dag([NodeSpec(id="stray", kind=NodeKind.synthesis, description="s",
                                      outputs={"report": "dict"})])
    connected = make_dag()
    responses = [disconnected, connected]
    prompts = []

    def fake_agent(model, system_prompt, user_content, schema, tools=None):
        prompts.append(user_content)
        return responses.pop(0), []

    monkeypatch.setattr(planner, "run_structured_agent", fake_agent)
    dag = planner.plan("demo task", model=None)
    assert dag is connected
    assert len(prompts) == 2
    assert "disconnected graphs" in prompts[1] and "Previous plan:" in prompts[1]


def test_plan_accepts_valid_dag_first_try(monkeypatch):
    connected = make_dag()
    calls = []

    def fake_agent(model, system_prompt, user_content, schema, tools=None):
        calls.append(user_content)
        return connected, []

    monkeypatch.setattr(planner, "run_structured_agent", fake_agent)
    assert planner.plan("demo task", model=None) is connected
    assert len(calls) == 1
