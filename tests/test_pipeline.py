"""Offline tests: registry nodes + contract testing, no LLM calls involved."""
import json

import geopandas as gpd

from llm_geo.coder import signature_for
from llm_geo.contracts import run_contract
from llm_geo.executor import execute
from llm_geo.models import DAGSpec, NodeKind, NodeSpec, PortSpec
from llm_geo.planner import validate_dag
from llm_geo.synthetic import make_inputs
from llm_geo.trace import Tracer

GOOD_CODE = """
def run(features) -> dict:
    return {"report": {"n": len(features)}}
"""

BAD_CODE = """
def run(features) -> dict:
    return "not a dict"
"""

KWARGS_CODE = """
def run(**inputs) -> dict:
    return {"report": {}}
"""


def synth_node() -> NodeSpec:
    return NodeSpec(
        id="summarize_x", kind=NodeKind.synthesis, description="count features",
        depends_on=["x"], inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"},
    )


def test_make_inputs_covers_declared_types():
    inputs = make_inputs({"a": "GeoDataFrame", "b": "float", "c": "str", "d": "bool", "e": "int"})
    assert isinstance(inputs["a"], gpd.GeoDataFrame)
    assert isinstance(inputs["b"], float)
    assert isinstance(inputs["c"], str)
    assert inputs["d"] is True
    assert isinstance(inputs["e"], int) and not isinstance(inputs["e"], bool)


def test_make_inputs_honors_port_constraints():
    port = PortSpec(type="GeoDataFrame", columns={"population": "int"}, geometry="Polygon", crs="EPSG:3857")
    gdf = make_inputs({"zones": port})["zones"]
    assert "population" in gdf.columns
    assert set(gdf.geom_type.unique()) == {"Polygon"}
    assert gdf.crs == "EPSG:3857"
    assert make_inputs({"n": PortSpec(type="int", example=7)})["n"] == 7
    # an example violating the declared type falls back to a synthetic value
    assert make_inputs({"n": PortSpec(type="int", example="seven")})["n"] == 2


def test_contract_pass_and_fail():
    node = synth_node()
    assert run_contract(node, GOOD_CODE).ok
    bad = run_contract(node, BAD_CODE)
    assert not bad.ok and "dict" in bad.error


def test_contract_rejects_kwargs_signature():
    result = run_contract(synth_node(), KWARGS_CODE)
    assert not result.ok and "explicit named parameters" in result.error


def test_contract_rejects_constraint_violations():
    node = NodeSpec(
        id="t", kind=NodeKind.transformation, description="t",
        inputs={"features": "GeoDataFrame"},
        outputs={"count": {"type": "int"}, "features": {"type": "GeoDataFrame", "columns": {"score": "float"}}},
    )
    float_count = """
def run(features) -> dict:
    return {"count": 1.0, "features": features.assign(score=0.5)}
"""
    missing_column = """
def run(features) -> dict:
    return {"count": 1, "features": features}
"""
    good = """
def run(features) -> dict:
    return {"count": 1, "features": features.assign(score=0.5)}
"""
    assert not run_contract(node, float_count).ok
    assert not run_contract(node, missing_column).ok
    assert run_contract(node, good).ok


def test_signature_for_includes_inputs_and_params():
    node = NodeSpec(
        id="t", kind=NodeKind.transformation, description="t",
        inputs={"features": "GeoDataFrame", "distance": "float"}, outputs={"features": "GeoDataFrame"},
        params={"distance": 10, "mode": "fast"},
    )
    assert signature_for(node) == "def run(features: gpd.GeoDataFrame, distance: float, mode='fast') -> dict:"


def test_validate_dag_catches_wiring_and_type_errors():
    dag = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.retrieval, description="a",
                 outputs={"features": "GeoDataFrame"}, registry_id=None),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"features": "dict", "threshold": "float"}, outputs={"report": "dict"}),
    ])
    errors = validate_dag(dag)
    assert any("produced as GeoDataFrame but consumed as dict" in e for e in errors)
    assert any("input 'threshold' has no source" in e for e in errors)


def test_validate_dag_checks_registry_contract():
    dag = DAGSpec(task="t", nodes=[
        NodeSpec(id="read", kind=NodeKind.retrieval, description="read", registry_id="read_geojson",
                 outputs={"features": "GeoDataFrame"}),  # missing required 'path'
        NodeSpec(id="buf", kind=NodeKind.transformation, description="buffer", registry_id="buffer",
                 depends_on=["read"], inputs={"features": "GeoDataFrame", "distance": "float"},
                 outputs={"features": "GeoDataFrame"}, params={"distance": 10}),
    ])
    errors = validate_dag(dag)
    assert any("requires input 'path'" in e for e in errors)
    assert not any("'buf'" in e for e in errors)


def test_registry_pipeline_end_to_end(tmp_path):
    geojson_path = tmp_path / "pts.geojson"
    geojson_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}}],
    }))

    dag = DAGSpec(task="buffer and summarize", nodes=[
        NodeSpec(id="read", kind=NodeKind.retrieval, description="read", registry_id="read_geojson",
                  inputs={}, outputs={"features": "GeoDataFrame"}, params={"path": str(geojson_path)}),
        NodeSpec(id="buf", kind=NodeKind.transformation, description="buffer", registry_id="buffer",
                  depends_on=["read"], inputs={"features": "GeoDataFrame", "distance": "float"},
                  outputs={"features": "GeoDataFrame"}, params={"distance": 10}),
        NodeSpec(id="rep", kind=NodeKind.synthesis, description="summarize", registry_id="summarize",
                  depends_on=["buf"], inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"}),
    ])

    assert validate_dag(dag) == []
    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert result.success, result.error
    assert result.outputs["rep"]["report"]["count"] == 1


def test_executor_blames_producer_on_edge_violation(tmp_path):
    dag = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.transformation, description="a", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"}),
    ])
    from llm_geo.models import NodeImplementation
    impls = {
        "a": NodeImplementation(node_id="a", code="def run() -> dict:\n    return {'features': {'not': 'a gdf'}}"),
        "b": NodeImplementation(node_id="b", code="def run(features) -> dict:\n    return {'report': {}}"),
    }
    result = execute(dag, impls, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
    assert result.failing_node_ids == ["a"]
    assert "produced by 'a'" in result.error


def test_cycle_is_detected(tmp_path):
    dag = DAGSpec(task="cyclic", nodes=[
        NodeSpec(id="a", kind=NodeKind.transformation, description="a", depends_on=["b"]),
        NodeSpec(id="b", kind=NodeKind.transformation, description="b", depends_on=["a"]),
    ])
    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
