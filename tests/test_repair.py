"""Offline tests for edit-based repair and partial re-execution: no LLM calls involved."""
import pytest

from llm_geo import coder
from llm_geo.coder import apply_edits, implement_node
from llm_geo.executor import execute
from llm_geo.models import CodeEdit, DAGSpec, NodeCodeEdits, NodeKind, NodeSpec, NodeImplementation
from llm_geo.trace import Tracer


def counting_code(marker_path: str, params: str, body: str) -> str:
    """Node code that appends one char to `marker_path` per call (our call counter)."""
    return (
        f"def run({params}) -> dict:\n"
        f"    open({marker_path!r}, 'a').write('x')\n"
        f"    {body}\n"
    )


def calls(path) -> int:
    return len(path.read_text()) if path.exists() else 0


def chain_dag() -> DAGSpec:
    return DAGSpec(task="demo", nodes=[
        NodeSpec(id="a", kind=NodeKind.retrieval, description="a", outputs={"value": "dict"}),
        NodeSpec(id="b", kind=NodeKind.transformation, description="b", depends_on=["a"],
                 inputs={"value": "dict"}, outputs={"result": "dict"}),
        NodeSpec(id="c", kind=NodeKind.synthesis, description="c", depends_on=["b"],
                 inputs={"result": "dict"}, outputs={"report": "dict"}),
    ])


def impls(a_code: str, b_code: str, c_code: str) -> dict[str, NodeImplementation]:
    return {
        "a": NodeImplementation(node_id="a", code=a_code),
        "b": NodeImplementation(node_id="b", code=b_code),
        "c": NodeImplementation(node_id="c", code=c_code),
    }


def test_partial_reexecution_reuses_prior_outputs(tmp_path):
    tracer = Tracer(path=tmp_path / "trace.jsonl")
    a_marker = tmp_path / "a_calls"
    dag = chain_dag()
    a_code = counting_code(str(a_marker), "", "return {'value': {'n': 1}}")
    bad_b = "def run(value: dict) -> dict:\n    raise ValueError('kaboom')\n"
    c_code = "def run(result: dict) -> dict:\n    return {'report': dict(result)}\n"

    first = execute(dag, impls(a_code, bad_b, c_code), tracer)
    assert not first.success and first.failing_node_ids == ["b"]
    assert calls(a_marker) == 1

    good_b = "def run(value: dict) -> dict:\n    return {'result': dict(value)}\n"
    second = execute(
        dag, impls(a_code, good_b, c_code), tracer,
        prior_outputs=first.outputs, stale_node_ids=frozenset(first.failing_node_ids),
    )
    assert second.success
    assert calls(a_marker) == 1  # `a` was NOT re-executed
    assert second.node_status == {"a": "cached", "b": "ok", "c": "ok"}
    assert second.node_duration_ms["a"] == 0.0
    assert second.outputs["c"] == {"report": {"n": 1}}


def test_prior_outputs_downstream_of_stale_are_rerun(tmp_path):
    tracer = Tracer(path=tmp_path / "trace.jsonl")
    c_marker = tmp_path / "c_calls"
    dag = chain_dag()
    a_code = "def run() -> dict:\n    return {'value': {'n': 2}}\n"
    b_code = "def run(value: dict) -> dict:\n    return {'result': dict(value)}\n"
    c_code = counting_code(str(c_marker), "result: dict", "return {'report': dict(result)}")

    # Prior outputs cover ALL nodes, but b is stale -> c (downstream) must re-run, a is cached.
    prior = {"a": {"value": {"n": 2}}, "b": {"result": {"n": 0}}, "c": {"report": {"n": 0}}}
    result = execute(dag, impls(a_code, b_code, c_code), tracer,
                     prior_outputs=prior, stale_node_ids=frozenset({"b"}))
    assert result.success
    assert result.node_status == {"a": "cached", "b": "ok", "c": "ok"}
    assert calls(c_marker) == 1
    assert result.outputs["c"] == {"report": {"n": 2}}  # fresh, not the stale prior value


def test_apply_edits():
    code = "def run(**inputs) -> dict:\n    return {'x': 1}\n"
    edited = apply_edits(code, [CodeEdit(find="{'x': 1}", replace="{'x': 2}")])
    assert "{'x': 2}" in edited
    with pytest.raises(ValueError, match="exactly once"):
        apply_edits(code, [CodeEdit(find="not in code", replace="y")])
    with pytest.raises(ValueError, match="exactly once"):
        apply_edits("aa", [CodeEdit(find="a", replace="b")])
    with pytest.raises(ValueError, match="no edits"):
        apply_edits(code, [])


def test_repair_via_edits(monkeypatch):
    node = NodeSpec(id="calc", kind=NodeKind.synthesis, description="calc",
                    inputs={"value": "dict"}, outputs={"report": "dict"})
    previous_code = "def run(value: dict) -> dict:\n    return {'report': value['missing_key']}\n"
    prompts = []

    def fake_agent(model, system_prompt, user_content, schema, tools=None):
        prompts.append((system_prompt, user_content))
        assert schema is NodeCodeEdits
        return NodeCodeEdits(node_id="calc", edits=[
            CodeEdit(find="value['missing_key']", replace="{'n': len(value)}"),
        ], notes="use available inputs"), []

    monkeypatch.setattr(coder, "run_structured_agent", fake_agent)
    impl, attempts = implement_node(node, model=None, repair_context={
        "previous_code": previous_code,
        "error": "KeyError: 'missing_key'",
        "traceback": "Traceback ...\nKeyError: 'missing_key'",
    })

    assert attempts == 1
    assert "missing_key" not in impl.code and "{'n': len(value)}" in impl.code
    system_prompt, user_content = prompts[0]
    assert "REPAIRING" in system_prompt
    assert previous_code.strip() in user_content  # previous code shown for editing
    assert "KeyError: 'missing_key'" in user_content  # runtime failure shown


def test_repair_recovers_from_bad_edits(monkeypatch):
    node = NodeSpec(id="calc", kind=NodeKind.synthesis, description="calc", outputs={"report": "dict"})
    previous_code = "def run() -> dict:\n    return {'wrong': 1}\n"
    responses = [
        NodeCodeEdits(node_id="calc", edits=[CodeEdit(find="does not exist", replace="x")]),
        NodeCodeEdits(node_id="calc", edits=[CodeEdit(find="{'wrong': 1}", replace="{'report': {}}")]),
    ]
    seen_feedback = []

    def fake_agent(model, system_prompt, user_content, schema, tools=None):
        seen_feedback.append(user_content)
        return responses.pop(0), []

    monkeypatch.setattr(coder, "run_structured_agent", fake_agent)
    impl, attempts = implement_node(node, model=None, repair_context={
        "previous_code": previous_code, "error": "boom", "traceback": "",
    })
    assert attempts == 2
    assert "{'report': {}}" in impl.code
    assert "could not be applied" in seen_feedback[1]


def test_implement_without_repair_context_unchanged(monkeypatch):
    node = NodeSpec(id="calc", kind=NodeKind.synthesis, description="calc", outputs={"report": "dict"})

    def fake_agent(model, system_prompt, user_content, schema, tools=None):
        assert schema is NodeImplementation
        return NodeImplementation(node_id="calc", code="def run() -> dict:\n    return {'report': {}}\n"), []

    monkeypatch.setattr(coder, "run_structured_agent", fake_agent)
    impl, attempts = implement_node(node, model=None)
    assert attempts == 1 and "report" in impl.code
