"""Validated, atomic OpenAPI operation generation."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from llm_geo.operations.openapi.classify import classify_operations
from llm_geo.operations.openapi.models import GenerationResult
from llm_geo.operations.openapi.parser import parse_openapi, python_identifier
from llm_geo.operations.openapi.renderer import render_module

GENERATOR_VERSION = 2  # bumped: @code now requires kind=, retrieval GeoJSON responses -> GeoDataFrame
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parents[1] / "generated"


def _spec_hash(spec: dict[str, Any]) -> str:
    serialized = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_generated_module(
    module_path: Path, module_name: str, expected_names: tuple[str, ...]
) -> None:
    """Import generated source in an isolated interpreter and inspect registration."""
    ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    validation_script = """
import importlib.util
import json
import sys

from llm_geo.operations import registered_operations

path, module_name, expected_json = sys.argv[1:]
spec = importlib.util.spec_from_file_location(module_name, path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load generated module at {path}")
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
actual = sorted(
    operation.name
    for operation in registered_operations()
    if operation.module == module_name
)
expected = sorted(json.loads(expected_json))
if actual != expected:
    raise RuntimeError(f"Registered operations differ: expected={expected}, actual={actual}")
"""
    environment = os.environ.copy()
    project_root = str(Path(__file__).parents[3])
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join([project_root, existing_path]) if existing_path else project_root
    )
    command = [
        sys.executable,
        "-c",
        validation_script,
        str(module_path),
        module_name,
        json.dumps(expected_names),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"Generated module validation failed: {detail}") from error


def generate_openapi_operations(
    spec: dict[str, Any],
    *,
    service: str,
    default_base_url: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    source: str = "openapi.json",
    api_key_environment: str | None = None,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    force: bool = False,
    classifier_model: BaseChatModel | None = None,
) -> GenerationResult:
    """Generate, validate, and atomically promote operations for one service."""
    normalized_spec = copy.deepcopy(spec)
    parsed = parse_openapi(normalized_spec)
    if not parsed.operations:
        reasons = "; ".join(item["reason"] for item in parsed.skipped)
        raise ValueError(f"OpenAPI document has no supported operations: {reasons}")
    service_name = python_identifier(service, "service")
    output_directory = Path(output_directory)
    specs_directory = output_directory / "specs"
    module_path = output_directory / f"{service_name}.py"
    spec_path = specs_directory / f"{service_name}.openapi.json"
    manifest_path = specs_directory / f"{service_name}.manifest.json"
    digest = _spec_hash(normalized_spec)
    operation_names = tuple(item.function_name for item in parsed.operations)
    if not force and module_path.exists() and spec_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if (
            manifest.get("spec_sha256") == digest
            and manifest.get("generator_version") == GENERATOR_VERSION
            and manifest.get("operation_ids") == list(operation_names)
        ):
            return GenerationResult(
                module_path, spec_path, manifest_path, operation_names, False
            )
    classifications = classify_operations(parsed.operations, classifier_model)
    kinds = {name: classification.kind for name, classification in classifications.items()}
    returns_geojson = {
        name: classification.returns_geojson for name, classification in classifications.items()
    }
    source_code = render_module(
        parsed.operations,
        service=service_name,
        default_base_url=default_base_url,
        kinds=kinds,
        returns_geojson=returns_geojson,
        api_key_environment=api_key_environment,
        auth_header=auth_header,
        auth_scheme=auth_scheme,
    )
    manifest = {
        "service": service_name,
        "source": source,
        "spec_sha256": digest,
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation_ids": list(operation_names),
        "classifications": {
            name: {"kind": kinds[name], "returns_geojson": returns_geojson[name]}
            for name in operation_names
        },
        "skipped": list(parsed.skipped),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    specs_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openapi-generation-", dir=output_directory) as temporary:
        staging = Path(temporary)
        staged_module = staging / module_path.name
        staged_spec = staging / spec_path.name
        staged_manifest = staging / manifest_path.name
        staged_module.write_text(source_code, encoding="utf-8")
        staged_spec.write_text(
            json.dumps(normalized_spec, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation_module = f"llm_geo.operations.generated._validate_{service_name}"
        _validate_generated_module(staged_module, validation_module, operation_names)
        os.replace(staged_module, module_path)
        os.replace(staged_spec, spec_path)
        os.replace(staged_manifest, manifest_path)
    return GenerationResult(module_path, spec_path, manifest_path, operation_names, True)
