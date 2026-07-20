"""Offline tests for MapNode fan-out: executor collection, planner validation, synthetic lists.

No LLM and no live registry op are involved: executor tests inject fake trusted ops into the shared
`registry.REGISTRY` dict, and planner-validation tests use the always-registered `geocode_place`.
"""
import geopandas as gpd
import pytest
from shapely.geometry import Point

from llm_geo import registry
from llm_geo.executor import MAX_MAP_ITEMS, execute
from llm_geo.models import DAGSpec, NodeKind, NodeSpec
from llm_geo.planner import map_errors
from llm_geo.synthetic import make_inputs, make_value
from llm_geo.trace import Tracer


@pytest.fixture
def fake_ops():
    """Inject fake trusted ops into the shared registry for the duration of one test."""

    def geocode(**kwargs) -> dict:
        query = kwargs["query"]
        gdf = gpd.GeoDataFrame({"name": [query]}, geometry=[Point(0.0, 0.0)], crs="EPSG:4326")
        gdf.attrs["provenance"] = {"query": query}
        return {"features": gdf}

    def measure(**kwargs) -> dict:
        return {"report": {"len": len(kwargs["text"])}}

    def boom(**kwargs) -> dict:
        if kwargs["q"] == "boom":
            raise ValueError("kaboom")
        return {"report": {"q": kwargs["q"]}}

    added = {
        "fake_geocode": {"inputs": {"query": "str"}, "fn": geocode},
        "fake_measure": {"inputs": {"text": "str"}, "fn": measure},
        "fake_boom": {"inputs": {"q": "str"}, "fn": boom},
    }
    registry.REGISTRY.update(added)
    try:
        yield
    finally:
        for key in added:
            registry.REGISTRY.pop(key, None)


def _map_node(**overrides) -> NodeSpec:
    base = dict(
        id="mapped", kind=NodeKind.retrieval, description="fan out",
        registry_id="fake_geocode", map_over="query",
        inputs={"query": "list[str]"}, outputs={"features": "GeoDataFrame"},
    )
    base.update(overrides)
    return NodeSpec(**base)


# ---- executor: collection semantics ----

def test_map_concatenates_geodataframes(fake_ops, tmp_path):
    dag = DAGSpec(task="geocode many", nodes=[_map_node(params={"query": ["a", "b", "c"]})])
    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert result.success, result.error
    features = result.outputs["mapped"]["features"]
    assert isinstance(features, gpd.GeoDataFrame)
    assert len(features) == 3
    assert features.attrs["provenance"] == [{"query": "a"}, {"query": "b"}, {"query": "c"}]


def test_map_collects_non_geodataframe_outputs_into_list(fake_ops, tmp_path):
    node = _map_node(
        registry_id="fake_measure", map_over="text",
        inputs={"text": "list[str]"}, outputs={"report": "list[dict]"},
        params={"text": ["ab", "cde"]},
    )
    result = execute(DAGSpec(task="measure many", nodes=[node]), {}, Tracer(tmp_path / "trace.jsonl"))
    assert result.success, result.error
    assert result.outputs["mapped"]["report"] == [{"len": 2}, {"len": 3}]


def test_map_empty_sequence_yields_empty_geodataframe(fake_ops, tmp_path):
    dag = DAGSpec(task="geocode none", nodes=[_map_node(params={"query": []})])
    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert result.success, result.error
    features = result.outputs["mapped"]["features"]
    assert isinstance(features, gpd.GeoDataFrame) and len(features) == 0


# ---- executor: failure handling ----

def test_map_item_failure_reports_index(fake_ops, tmp_path):
    node = _map_node(
        registry_id="fake_boom", map_over="q",
        inputs={"q": "list[str]"}, outputs={"report": "list[dict]"},
        params={"q": ["ok", "boom", "later"]},
    )
    result = execute(DAGSpec(task="boom", nodes=[node]), {}, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
    assert result.failing_node_ids == ["mapped"]
    assert "map item 2/3" in result.error and "kaboom" in result.error


def test_map_over_cap_is_enforced(fake_ops, tmp_path):
    node = _map_node(params={"query": [str(i) for i in range(MAX_MAP_ITEMS + 1)]})
    result = execute(DAGSpec(task="too many", nodes=[node]), {}, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
    assert f"cap of {MAX_MAP_ITEMS}" in result.error


def test_non_list_mapped_value_fails(fake_ops, tmp_path):
    node = _map_node(params={"query": "not-a-list"})
    result = execute(DAGSpec(task="bad", nodes=[node]), {}, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
    assert "must resolve to a list" in result.error


# ---- planner validation (map_errors) ----

def test_map_valid_plan_has_no_errors():
    node = NodeSpec(
        id="geo", kind=NodeKind.retrieval, description="geocode each", registry_id="geocode_place",
        map_over="query", inputs={"query": "list[str]"}, outputs={"features": "GeoDataFrame"},
    )
    assert map_errors(DAGSpec(task="t", nodes=[node])) == []


def test_map_requires_registry_id():
    node = NodeSpec(
        id="geo", kind=NodeKind.transformation, description="x", map_over="query",
        inputs={"query": "list[str]"}, outputs={"features": "GeoDataFrame"},
    )
    errors = map_errors(DAGSpec(task="t", nodes=[node]))
    assert any("no registry_id" in e for e in errors)


def test_map_over_must_be_list_typed():
    node = NodeSpec(
        id="geo", kind=NodeKind.retrieval, description="x", registry_id="geocode_place",
        map_over="query", inputs={"query": "str"}, outputs={"features": "GeoDataFrame"},
    )
    errors = map_errors(DAGSpec(task="t", nodes=[node]))
    assert any("not a list" in e for e in errors)


def test_map_over_must_be_a_declared_input():
    node = NodeSpec(
        id="geo", kind=NodeKind.retrieval, description="x", registry_id="geocode_place",
        map_over="query", inputs={"other": "list[str]"}, outputs={"features": "GeoDataFrame"},
    )
    errors = map_errors(DAGSpec(task="t", nodes=[node]))
    assert any("not one of its inputs" in e for e in errors)


def test_map_over_must_be_accepted_by_registry_op():
    node = NodeSpec(
        id="geo", kind=NodeKind.retrieval, description="x", registry_id="geocode_place",
        map_over="city", inputs={"city": "list[str]"}, outputs={"features": "GeoDataFrame"},
    )
    errors = map_errors(DAGSpec(task="t", nodes=[node]))
    assert any("does not accept" in e for e in errors)


# ---- synthetic list support ----

def test_make_value_builds_typed_lists():
    strings = make_value("list[str]")
    assert isinstance(strings, list) and len(strings) == 2 and all(isinstance(s, str) for s in strings)
    frames = make_value("list[GeoDataFrame]")
    assert all(isinstance(f, gpd.GeoDataFrame) for f in frames)


def test_make_inputs_covers_list_types():
    inputs = make_inputs({"queries": "list[str]", "frame": "GeoDataFrame"})
    assert isinstance(inputs["queries"], list) and inputs["queries"] == ["synthetic-string", "synthetic-string"]
    assert isinstance(inputs["frame"], gpd.GeoDataFrame)
