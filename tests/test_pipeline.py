"""Offline tests: registry nodes + contract testing, no LLM calls involved."""
import json

import geopandas as gpd

from llm_geo.contracts import run_contract
from llm_geo.executor import execute
from llm_geo.models import DAGSpec, NodeKind, NodeSpec
from llm_geo.synthetic import make_inputs
from llm_geo.trace import Tracer

GOOD_CODE = """
def run(**inputs) -> dict:
    return {"report": {"n": len(inputs["features"])}}
"""

BAD_CODE = """
def run(**inputs) -> dict:
    return "not a dict"
"""


def synth_node() -> NodeSpec:
    return NodeSpec(
        id="summarize_x", kind=NodeKind.synthesis, description="count features",
        depends_on=["x"], inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"},
    )


def test_make_inputs_covers_declared_types():
    inputs = make_inputs({"a": "GeoDataFrame", "b": "float", "c": "str", "d": "bool"})
    assert isinstance(inputs["a"], gpd.GeoDataFrame)
    assert isinstance(inputs["b"], float)
    assert isinstance(inputs["c"], str)
    assert inputs["d"] is True


def test_contract_pass_and_fail():
    node = synth_node()
    assert run_contract(node, GOOD_CODE).ok
    bad = run_contract(node, BAD_CODE)
    assert not bad.ok and "dict" in bad.error


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

    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert result.success, result.error
    assert result.outputs["rep"]["report"]["count"] == 1


def test_cycle_is_detected(tmp_path):
    dag = DAGSpec(task="cyclic", nodes=[
        NodeSpec(id="a", kind=NodeKind.transformation, description="a", depends_on=["b"]),
        NodeSpec(id="b", kind=NodeKind.transformation, description="b", depends_on=["a"]),
    ])
    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
