"""End-to-end demonstration of the agentic LLM geo-analysis workflow (plan -> implement/validate
-> assemble/execute -> trace), exercising both local-file and live OSM retrieval nodes.

Uses the OpenAI API key from the environment (.env is loaded if present).

Run with: python main.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY not found in the environment (set it, or add it to a .env file).")

from llm_geo.artifacts import RunArtifacts  # noqa: E402 (import after the API key is confirmed present)
from llm_geo.graph import run  # noqa: E402
from llm_geo.report import full_report, linear_order, terminal_output  # noqa: E402

DATA = Path(__file__).parent / "data"
REPORTS = Path(__file__).parent / "reports"


def p(name: str) -> str:
    return (DATA / name).resolve().as_posix()


TEST_CASES = [
    {
        "name": "buffer_and_summarize_points",
        "task": f"Read the GeoJSON points at {p('sample_points.geojson')}, buffer each by 100 meters, "
        "and summarize the resulting features (count, bounds, geometry types).",
        "expect_success": True,
    },
    {
        "name": "reproject_and_summarize",
        "task": f"Read the GeoJSON points at {p('sample_points.geojson')}, reproject them to EPSG:3857, "
        "and summarize the reprojected features.",
        "expect_success": True,
    },
    {
        "name": "filter_points_by_region_mask",
        "task": f"Read the GeoJSON points at {p('sample_points.geojson')} and the mask polygon at "
        f"{p('sample_region.geojson')}, keep only points that intersect the mask, and summarize them.",
        "expect_success": True,
    },
    {
        "name": "filter_parks_by_region_mask",
        "task": f"Read the park polygons at {p('sample_parks.geojson')} and the mask polygon at "
        f"{p('sample_region.geojson')}, keep only parks that intersect the mask, and summarize them.",
        "expect_success": True,
    },
    {
        "name": "geocode_landmark_nominatim",
        "task": "Geocode 'Golden Gate Bridge, San Francisco' using Nominatim and summarize the resulting "
        "features (count, bounds, geometry types).",
        "expect_success": True,
    },
    {
        "name": "overpass_cafes_near_downtown_sf",
        "task": "Query the Overpass API for cafes (amenity=cafe) inside the bbox "
        "'37.77,-122.45,37.80,-122.40' and summarize the resulting features.",
        "expect_success": True,
    },
    {
        "name": "roads_area_custom_synthesis",
        "task": f"Read the road LineStrings at {p('sample_roads.geojson')}, buffer them by 30 meters, "
        "reproject to EPSG:3857, then compute the total buffered polygon area in square meters and "
        "return a synthesis report with keys total_area_m2 and feature_count.",
        "expect_success": True,
    },
    {
        "name": "convex_hull_custom_synthesis",
        "task": f"Read the GeoJSON points at {p('sample_points.geojson')}, buffer each by 200 meters "
        "(no reprojection), compute the convex hull polygon enclosing the union of all buffered "
        "features, and return it as a geojson FeatureCollection under output key 'hull'.",
        "expect_success": True,
    },
    {
        "name": "geocode_then_buffer_and_summarize",
        "task": "Geocode 'Eiffel Tower, Paris' using Nominatim, buffer the result by 500 meters "
        "(no reprojection), and summarize count and bounds of the buffered features.",
        "expect_success": True,
    },
    {
        "name": "missing_file_graceful_failure",
        "task": f"Read the GeoJSON file at {p('does_not_exist.geojson')} and summarize it.",
        "expect_success": False,
    },
    # --- more complex cases below: multi-source fan-in, deeper chains, and several that force
    # --- custom LLM-generated nodes (haversine distance, area/union/centroid, density, overlay) ---
    {
        "name": "nearest_cafe_to_landmark",
        "task": "Geocode 'Dolores Park, San Francisco' using Nominatim, query the Overpass API for cafes "
        "(amenity=cafe) inside the bbox '37.75,-122.44,37.77,-122.42', then compute the great-circle "
        "(haversine) distance in meters from the geocoded point to each cafe, and return a synthesis "
        "report with the nearest cafe's name, its distance_m, and the total number of cafes considered.",
        "expect_success": True,
    },
    {
        "name": "roads_total_length",
        "task": f"Read the road LineStrings at {p('sample_roads.geojson')}, reproject to EPSG:3857, then "
        "compute the total length in meters of all road segments combined, and return a synthesis "
        "report with keys total_length_m and segment_count.",
        "expect_success": True,
    },
    {
        "name": "parks_union_area_and_centroid",
        "task": f"Read the park polygons at {p('sample_parks.geojson')}, reproject to EPSG:3857, then "
        "compute the union of all park polygons, its total area in square meters, and the centroid of "
        "that union reprojected back to EPSG:4326, returning a synthesis report with keys total_area_m2, "
        "centroid_lon and centroid_lat.",
        "expect_success": True,
    },
    {
        "name": "points_nearest_neighbor_distances",
        "task": f"Read the GeoJSON points at {p('sample_points.geojson')}, reproject to EPSG:3857, then "
        "for each point compute the distance in meters to its nearest other point, and return a synthesis "
        "report listing each point's name with its nearest_distance_m, plus the overall minimum distance "
        "across all points as min_distance_m.",
        "expect_success": True,
    },
    {
        "name": "deep_chain_buffered_points_in_parks",
        "task": f"Read the GeoJSON points at {p('sample_points.geojson')}, buffer each by 300 meters and "
        f"reproject the buffered points to EPSG:3857. Separately, read the park polygons at "
        f"{p('sample_parks.geojson')} and reproject them to EPSG:3857 too. Keep only parks that intersect "
        "the buffered points, and summarize the matched parks (count, bounds, geometry types).",
        "expect_success": True,
    },
    {
        "name": "restaurant_density_downtown_sf",
        "task": "Query the Overpass API for restaurants (amenity=restaurant) inside the bbox "
        "'37.76,-122.44,37.80,-122.40', then compute the point density in features per square kilometer "
        "within that bounding box, and return a synthesis report with keys count and density_per_km2.",
        "expect_success": True,
    },
    {
        "name": "distance_between_two_landmarks",
        "task": "Geocode 'San Francisco City Hall' using Nominatim as one retrieval node, and separately "
        "geocode 'Golden Gate Bridge, San Francisco' using Nominatim as an independent second retrieval "
        "node. Then compute the great-circle distance in kilometers between the two geocoded points, and "
        "return a synthesis report with key distance_km.",
        "expect_success": True,
    },
    {
        "name": "roads_park_overlap_area",
        "task": f"Read the road LineStrings at {p('sample_roads.geojson')}, buffer them by 20 meters and "
        f"reproject to EPSG:3857. Separately, read the park polygons at {p('sample_parks.geojson')} and "
        "reproject them to EPSG:3857 too. Then compute the total geometric overlap area in square meters "
        "between the buffered roads and the parks, and return a synthesis report with key overlap_area_m2.",
        "expect_success": True,
    },
    {
        "name": "cafes_convex_hull_service_area",
        "task": "Query the Overpass API for cafes (amenity=cafe) inside the bbox "
        "'37.77,-122.45,37.80,-122.40', then compute the convex hull polygon enclosing all the cafe "
        "locations, and return it as a geojson FeatureCollection under output key 'hull', along with the "
        "number of cafes used under key 'cafe_count' in the same synthesis output.",
        "expect_success": True,
    },
    {
        "name": "landmark_park_and_cafe_stress_test",
        "task": "Geocode 'Golden Gate Bridge, San Francisco' using Nominatim, query the Overpass API for "
        "cafes (amenity=cafe) inside the bbox '37.77,-122.48,37.82,-122.42', read the park polygons at "
        f"{p('sample_parks.geojson')} and the mask polygon at {p('sample_region.geojson')} and keep only "
        "parks intersecting the mask, buffer the geocoded landmark point by 2000 meters, then compute how "
        "many of the mask-filtered parks fall within that buffered landmark zone. Return one final "
        "synthesis report combining matched_park_count and the cafe_count found near the landmark.",
        "expect_success": True,
    },
]


def run_case(case: dict):
    """Returns (ok, detail, report, artifacts). `report` is None if the pipeline crashed outright;
    `artifacts` always points at the run's debug bundle (output/<case name>/<timestamp>/)."""
    artifacts = RunArtifacts(case["name"])
    try:
        report = run(case["task"], artifacts=artifacts)
    except Exception as exc:  # planner/coder crashed instead of reporting a graceful failure
        return False, f"raised {type(exc).__name__}: {exc}", None, artifacts

    result = report.result
    if case["expect_success"]:
        return result.success, ("ok" if result.success else f"pipeline failed: {result.error}"), report, artifacts
    ok = not result.success and bool(result.error)
    detail = "gracefully reported failure as expected" if ok else "expected a graceful failure but did not get one"
    return ok, detail, report, artifacts


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _flat_output(report) -> str:
    if not report or not report.result.outputs:
        return "-"
    _, out = terminal_output(report)
    return _truncate(json.dumps(out, default=str), 88)


def print_final_summary(cases: list[dict]) -> None:
    passed = sum(1 for c in cases if c["ok"])
    failed = len(cases) - passed

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for i, c in enumerate(cases, 1):
        report = c["report"]
        print(f"\n[{i:02d}] {c['name']} ... {'PASS' if c['ok'] else 'FAIL'}  ({c['elapsed']:.1f}s)")
        print(f"     input : {_truncate(c['task'], 88)}")
        print(f"     output: {_flat_output(report)}")
        if report:
            print(f"     graph : {linear_order(report)}")
            sources = [f"{n.id}={'registry' if n.registry_id else 'llm'}" for n in report.dag.nodes]
            print(f"     meta  : nodes={len(report.dag.nodes)} repair_rounds={report.repair_attempts} [{', '.join(sources)}]")
        else:
            print(f"     detail: {c['detail']}")
        print(f"     bundle: {c['artifacts'].dir}")

    print("\n" + "-" * 88)
    print(f"Total cases: {len(cases)}  Passed: {passed}  Failed: {failed}")
    print("Overall status:", "ALL PASSED" if failed == 0 else "FAILURES PRESENT")
    print("=" * 88)


def write_markdown_report(cases: list[dict]) -> Path:
    REPORTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = REPORTS / f"run_{stamp}.md"
    run_at = dt.datetime.now().isoformat(timespec="seconds")
    out_path.write_text(full_report(run_at, cases), encoding="utf-8")
    return out_path


def main() -> None:
    name_width = max(len(c["name"]) for c in TEST_CASES)
    cases: list[dict] = []

    print(f"Running {len(TEST_CASES)} end-to-end agentic workflow test cases...\n")
    for i, case in enumerate(TEST_CASES, 1):
        t0 = time.monotonic()
        ok, detail, report, artifacts = run_case(case)
        elapsed = time.monotonic() - t0
        print(f"[{i:02d}/{len(TEST_CASES)}] {'PASS' if ok else 'FAIL':<4} {case['name']:<{name_width}} "
              f"({elapsed:5.1f}s) - {detail}")
        if not ok:
            print(f"{'':>10}debug bundle: {artifacts.dir}")
        cases.append({**case, "ok": ok, "detail": detail, "report": report, "artifacts": artifacts, "elapsed": elapsed})
        if i < len(TEST_CASES):
            time.sleep(3)  # spread OpenAI token usage across the per-minute rate-limit window

    print_final_summary(cases)
    report_path = write_markdown_report(cases)
    print(f"\nFull report (input/output/solution graph/execution graph per case): {report_path}")


if __name__ == "__main__":
    main()
