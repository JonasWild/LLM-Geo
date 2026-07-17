"""Offline tests: port specs, contracts, synthetic inputs and executor validation. No LLM calls."""
import geopandas as gpd
from shapely.geometry import Point

from llm_geo import registry
from llm_geo.contracts import run_contract
from llm_geo.executor import execute
from llm_geo.models import DAGSpec, FieldSpec, NodeImplementation, NodeKind, NodeSpec, PortSpec
from llm_geo.synthetic import make_inputs
from llm_geo.trace import Tracer

GOOD_CODE = """
import geopandas as gpd
from typing_extensions import TypedDict

class Output(TypedDict):
    report: dict

def run(features: gpd.GeoDataFrame) -> Output:
    return {"report": {"n": len(features)}}
"""

BAD_RETURN_CODE = """
import geopandas as gpd

def run(features: gpd.GeoDataFrame) -> dict:
    return "not a dict"
"""

LEGACY_KWARGS_CODE = """
def run(**inputs) -> dict:
    return {"report": {"n": len(inputs["features"])}}
"""


def port(type_name: str, **kwargs) -> PortSpec:
    return PortSpec(type=type_name, description="test port", **kwargs)


def synth_node() -> NodeSpec:
    return NodeSpec(
        id="summarize_x", kind=NodeKind.synthesis, description="count features",
        depends_on=["x"], inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"},
    )


def test_legacy_string_ports_are_coerced_to_port_specs():
    node = synth_node()
    assert node.inputs["features"].type == "GeoDataFrame"
    assert node.outputs["report"].type == "dict"
    assert node.inputs["features"].description


def test_make_inputs_covers_declared_types():
    inputs = make_inputs({
        "a": port("GeoDataFrame"), "b": port("float"), "c": port("str"),
        "d": port("bool"), "e": port("int"), "f": port("dict"),
    })
    assert isinstance(inputs["a"], gpd.GeoDataFrame)
    assert isinstance(inputs["b"], float)
    assert isinstance(inputs["c"], str)
    assert inputs["d"] is True
    assert isinstance(inputs["e"], int) and not isinstance(inputs["e"], bool)
    assert isinstance(inputs["f"], dict)


def test_make_inputs_prefers_planner_examples():
    inputs = make_inputs({
        "crs": port("str", example="EPSG:25832"),
        "radius": port("float", example=3),  # int example is coerced to float
        "filters": port("dict", example={"min_capacity": 10}),
    })
    assert inputs["crs"] == "EPSG:25832"
    assert inputs["radius"] == 3.0 and isinstance(inputs["radius"], float)
    assert inputs["filters"] == {"min_capacity": 10}


def test_make_inputs_builds_dicts_from_declared_fields():
    spec = port("dict", fields={
        "count": FieldSpec(type="int", description="n"),
        "names": FieldSpec(type="list[str]", description="names"),
        "nested": FieldSpec(type="dict", description="free-form leaf"),
    })
    value = make_inputs({"report": spec})["report"]
    assert isinstance(value["count"], int) and not isinstance(value["count"], bool)
    assert isinstance(value["names"], list) and all(isinstance(x, str) for x in value["names"])
    assert isinstance(value["nested"], dict)


def test_fields_are_sanitized_on_the_port():
    # fields on a non-dict port are dropped
    assert port("str", fields={"x": FieldSpec(type="int", description="x")}).fields is None
    # an example conflicting with the field contract is dropped; a matching one survives
    fields = {"count": FieldSpec(type="int", description="n")}
    assert port("dict", fields=fields, example={"total": 3}).example is None
    assert port("dict", fields=fields, example={"count": 3}).example == {"count": 3}


def test_incoherent_example_is_dropped_not_fatal():
    assert port("int", example="abc").example is None
    assert port("int", example=True).example is None
    assert port("GeoDataFrame", example={"x": 1}).example is None
    assert port("float", example=3).example == 3


def test_contract_pass_and_fail():
    node = synth_node()
    assert run_contract(node, GOOD_CODE).ok
    bad = run_contract(node, BAD_RETURN_CODE)
    assert not bad.ok and "dict" in bad.error


def test_contract_rejects_kwargs_signature():
    result = run_contract(synth_node(), LEGACY_KWARGS_CODE)
    assert not result.ok and "*args/**kwargs" in result.error


def test_contract_enforces_names_annotations_and_return():
    wrong_name = "def run(stuff: dict) -> dict:\n    return {'report': {}}\n"
    result = run_contract(synth_node(), wrong_name)
    assert not result.ok and "missing parameters ['features']" in result.error

    wrong_annotation = "def run(features: dict) -> dict:\n    return {'report': {}}\n"
    result = run_contract(synth_node(), wrong_annotation)
    assert not result.ok and "declares it as GeoDataFrame" in result.error

    no_return = "import geopandas as gpd\ndef run(features: gpd.GeoDataFrame):\n    return {'report': {}}\n"
    result = run_contract(synth_node(), no_return)
    assert not result.ok and "return annotation" in result.error


def test_self_declared_return_typed_dict_is_enforced():
    # Coarse type (dict) is satisfied; only the coder's own TypedDict catches the bad value.
    code = """
import geopandas as gpd
from typing_extensions import TypedDict

class Output(TypedDict):
    report: dict[str, int]

def run(features: gpd.GeoDataFrame) -> Output:
    return {"report": {"count": "not-a-number"}}
"""
    result = run_contract(synth_node(), code)
    assert not result.ok and "count" in result.error


def test_output_dict_fields_are_enforced_key_by_key():
    node = NodeSpec(
        id="n", kind=NodeKind.synthesis, description="d",
        inputs={"features": "GeoDataFrame"},
        outputs={"report": port("dict", fields={
            "count": FieldSpec(type="int", description="n"),
            "names": FieldSpec(type="list[str]", description="names"),
        })},
    )

    def code(returned: str) -> str:
        return (
            "import geopandas as gpd\n"
            "def run(features: gpd.GeoDataFrame) -> dict:\n"
            f"    return {{'report': {returned}}}\n"
        )

    missing_key = run_contract(node, code("{'total': 1, 'names': ['a']}"))
    assert not missing_key.ok and "missing declared key 'count'" in missing_key.error

    wrong_type = run_contract(node, code("{'count': '1', 'names': ['a']}"))
    assert not wrong_type.ok and "key 'count' must be int, got str" in wrong_type.error

    assert run_contract(node, code("{'count': 1, 'names': ['a', 'b']}")).ok


def test_geodataframe_inside_return_typed_dict_validates():
    # Regression: a TypedDict defined in exec'd node code defers pydantic's schema build;
    # without an explicit rebuild the fine validator silently disappeared.
    code = """
import geopandas as gpd
from shapely.geometry import Point
from typing_extensions import TypedDict

class Output(TypedDict):
    features: gpd.GeoDataFrame

def run() -> Output:
    return {"features": gpd.GeoDataFrame({"name": ["x"]}, geometry=[Point(0, 0)], crs="EPSG:4326")}
"""
    node = NodeSpec(id="t", kind=NodeKind.transformation, description="d",
                    outputs={"features": "GeoDataFrame"})
    assert run_contract(node, code).ok


def test_params_are_passed_with_real_literal_values():
    node = NodeSpec(id="p", kind=NodeKind.transformation, description="scale",
                    params={"distance": 10}, outputs={"result": "dict"})
    code = (
        "def run(distance: int) -> dict:\n"
        "    assert distance == 10, distance\n"
        "    return {'result': {'distance': distance}}\n"
    )
    assert run_contract(node, code).ok


def retrieval_code(with_provenance: bool) -> str:
    attrs = "gdf.attrs['provenance'] = {'source': 'test'}\n    " if with_provenance else ""
    return (
        "import geopandas as gpd\n"
        "from shapely.geometry import Point\n\n"
        "def run() -> dict:\n"
        "    gdf = gpd.GeoDataFrame({'name': ['x']}, geometry=[Point(0, 0)], crs='EPSG:4326')\n"
        f"    {attrs}return {{'features': gdf}}\n"
    )


def test_retrieval_geodataframe_requires_provenance():
    node = NodeSpec(id="r", kind=NodeKind.retrieval, description="fetch",
                    outputs={"features": "GeoDataFrame"})
    missing = run_contract(node, retrieval_code(False))
    assert not missing.ok and "provenance" in missing.error
    assert run_contract(node, retrieval_code(True)).ok


def _points_with_provenance() -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0.0, 0.0), Point(1.0, 1.0)], crs="EPSG:4326",
    )
    gdf.attrs["provenance"] = {"source": "test"}
    return gdf


def test_pipeline_with_registry_and_custom_nodes(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "make_points", {
        "kind": "retrieval", "description": "two demo points",
        "inputs": {}, "outputs": {"features": "GeoDataFrame"},
        "fn": lambda **kwargs: {"features": _points_with_provenance()},
    })
    dag = DAGSpec(task="count points", nodes=[
        NodeSpec(id="fetch", kind=NodeKind.retrieval, description="fetch",
                 registry_id="make_points", outputs={"features": "GeoDataFrame"}),
        NodeSpec(id="rep", kind=NodeKind.synthesis, description="summarize", depends_on=["fetch"],
                 inputs={"features": "GeoDataFrame"}, outputs={"report": "dict"}),
    ])
    impls = {"rep": NodeImplementation(node_id="rep", code=GOOD_CODE)}
    result = execute(dag, impls, Tracer(tmp_path / "trace.jsonl"))
    assert result.success, result.error
    assert result.outputs["rep"]["report"]["n"] == 2


def test_execution_output_violation_is_attributed_to_producer(tmp_path):
    dag = DAGSpec(task="t", nodes=[
        NodeSpec(id="a", kind=NodeKind.transformation, description="a", outputs={"value": "dict"}),
        NodeSpec(id="b", kind=NodeKind.synthesis, description="b", depends_on=["a"],
                 inputs={"value": "dict"}, outputs={"report": "dict"}),
    ])
    bad_a = "def run() -> dict:\n    return {'value': 3}\n"  # declared dict, returns int
    good_b = "def run(value: dict) -> dict:\n    return {'report': dict(value)}\n"
    impls = {"a": NodeImplementation(node_id="a", code=bad_a),
             "b": NodeImplementation(node_id="b", code=good_b)}
    result = execute(dag, impls, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success and result.failing_node_ids == ["a"]
    assert "output contract of node 'a' violated" in result.error
    assert "'value' must be dict" in result.error


def test_cycle_is_detected(tmp_path):
    dag = DAGSpec(task="cyclic", nodes=[
        NodeSpec(id="a", kind=NodeKind.transformation, description="a", depends_on=["b"]),
        NodeSpec(id="b", kind=NodeKind.transformation, description="b", depends_on=["a"]),
    ])
    result = execute(dag, {}, Tracer(tmp_path / "trace.jsonl"))
    assert not result.success
