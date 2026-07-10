#!/usr/bin/env python3
"""
Generate a typed openapi-python-client package from an MCPO OpenAPI schema,
then generate a tiny facade file with one Python function per REST endpoint.

Usage:
  # CLI:
  export MCPO_OPENAPI_URL="http://localhost:8000/openapi.json"
  export MCPO_API_KEY="top-secret"   # optional, but normally needed for mcpo
  python generate_mcpo_client_and_facade.py

  # Programmatic:
  from generate_mcpo_client_and_facade import generate_mcpo_client
  generate_mcpo_client(
      openapi_url="http://localhost:8000/openapi.json",
      generated_projects_dir="./generated",
      package_name="mcpo_client",
      api_key="top-secret"
  )

Outputs:
  <generated_projects_dir>/openapi.json                        # downloaded OpenAPI spec
  <generated_projects_dir>/openapi-python-client-config.yaml   # generation config
  <generated_projects_dir>/<package_name>/                     # generated client package
  <generated_projects_dir>/mcpo_functions.py                   # clean facade with @code decorators
"""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SPEC_PATH = Path(os.getenv("MCPO_OPENAPI_PATH", "openapi.json"))
DEFAULT_CONFIG_PATH = Path("openapi-python-client-config.yaml")
DEFAULT_GENERATED_PROJECT_DIR = Path("generated/mcpo_client")
DEFAULT_PACKAGE_NAME = "mcpo_client"
DEFAULT_FACADE_PATH = Path("generated/mcpo_functions.py")

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

# Names that should stay unqualified in generated type annotations/defaults.
# Everything else in operation signatures is usually imported inside the generated
# operation module, so the facade references it as _op_x.Name.
SAFE_ANNOTATION_NAMES = {
    "Any",
    "None",
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
}

OPENAPI_SERVERS = [
    {
        "package_name": "geo_mcp",
        "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194/openapi.json",
        "api_key": None,
        "usage_postfix": "Usage: df = gpd.GeoDataFrame().from_dict(result)",
    },
]

BASE_DIR = Path(__file__).parent.parent / "function_library" / "openapi"


@dataclass(frozen=True)
class OperationModule:
    facade_name: str
    module_import_path: str
    alias: str
    path: Path


def snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value:
        value = "endpoint"
    if value[0].isdigit():
        value = f"endpoint_{value}"
    if keyword.iskeyword(value):
        value = f"{value}_"
    return value


def clean_function_name(name: str) -> str:
    """Remove tool_ prefix and _post suffix from function names."""
    if name.startswith("tool_"):
        name = name[5:]
    if name.endswith("_post"):
        name = name[:-5]
    return name


def fallback_operation_id(method: str, path: str) -> str:
    path_name = path.replace("{", "by_").replace("}", "")
    return snake_case(f"{method}_{path_name}")


def normalize_operation_ids(spec: dict[str, Any]) -> None:
    """Make operationId values stable, Pythonic, and unique before generation."""
    used: set[str] = set()
    for path, path_item in sorted(spec.get("paths", {}).items()):
        if not isinstance(path_item, dict):
            continue
        for method, operation in sorted(path_item.items()):
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            base = snake_case(
                str(operation.get("operationId") or fallback_operation_id(method, path))
            )
            name = base
            counter = 2
            while name in used:
                name = f"{base}_{counter}"
                counter += 1
            used.add(name)
            operation["operationId"] = name


def compute_spec_hash(spec: dict[str, Any]) -> str:
    """Compute SHA256 hash of normalized OpenAPI spec."""
    spec_json = json.dumps(spec, indent=2, sort_keys=True)
    return hashlib.sha256(spec_json.encode("utf-8")).hexdigest()


def fetch_openapi_spec(openapi_url: str) -> dict[str, Any]:
    """Fetch OpenAPI spec from URL and return as dict."""
    print(f"Fetching OpenAPI schema from {openapi_url}")
    with urllib.request.urlopen(openapi_url, timeout=30) as response:
        spec = json.loads(response.read().decode("utf-8"))
    normalize_operation_ids(spec)
    return spec


def get_saved_spec_path(spec_path: Path) -> Path:
    """Get path to saved spec file for comparison."""
    return spec_path.parent / f"saved_{spec_path.name}"


def specs_are_sync(current_spec: dict[str, Any], saved_spec_path: Path) -> bool:
    """Check if current spec matches saved spec.

    Args:
        current_spec: Current OpenAPI spec dict
        saved_spec_path: Path to saved spec file

    Returns:
        True if specs match, False otherwise
    """
    if not saved_spec_path.exists():
        return False

    try:
        saved_spec = json.loads(saved_spec_path.read_text(encoding="utf-8"))
        current_hash = compute_spec_hash(current_spec)
        saved_hash = compute_spec_hash(saved_spec)
        return current_hash == saved_hash
    except Exception as e:
        print(f"Warning: Could not compare specs: {e}")
        return False


def save_spec_for_comparison(spec: dict[str, Any], saved_spec_path: Path) -> None:
    """Save spec to file for future comparison."""
    saved_spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved spec to {saved_spec_path} for future comparison")


def check_and_generate_if_needed(
    openapi_url: str,
    generated_projects_dir: Path,
    package_name: str = DEFAULT_PACKAGE_NAME,
    api_key: str | None = None,
    usage_postfix: str | None = None,
) -> tuple[Path, Path, bool]:
    """Check if OpenAPI spec has changed and regenerate if needed.

    Args:
        openapi_url: URL to OpenAPI JSON schema
        generated_projects_dir: Directory to generate client code into
        package_name: Name for the generated Python package
        api_key: Optional API key for authentication
        usage_postfix: Optional usage instructions appended to wrapper docstrings

    Returns:
        Tuple of (generated_client_dir, facade_file_path, was_regenerated)
    """
    generated_projects_dir_path = Path(generated_projects_dir)
    spec_path = generated_projects_dir_path / "openapi.json"
    saved_spec_path = get_saved_spec_path(spec_path)

    # Fetch current spec
    current_spec = fetch_openapi_spec(openapi_url)

    # Check if specs are in sync
    if specs_are_sync(current_spec, saved_spec_path):
        print(f"OpenAPI spec for {package_name} is up to date, skipping regeneration")
        # Return existing paths if they exist
        generated_project_dir = generated_projects_dir_path / package_name
        facade_path = generated_projects_dir_path / f"{package_name}_functions.py"
        if generated_project_dir.exists() and facade_path.exists():
            return generated_project_dir, facade_path, False
        # If files don't exist, we need to generate even though specs match
        print("Generated files not found, regenerating...")

    print(f"OpenAPI spec for {package_name} has changed, regenerating client...")

    # Generate new client
    generated_project_dir, facade_path = generate_mcpo_client(
        openapi_url=openapi_url,
        generated_projects_dir=str(generated_projects_dir_path),
        package_name=package_name,
        api_key=api_key,
        usage_postfix=usage_postfix,
        skip_sync_check=True,
    )

    # Save spec after successful generation
    save_spec_for_comparison(current_spec, saved_spec_path)

    return generated_project_dir, facade_path, True


def fetch_and_prepare_spec(
    openapi_url: str,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> None:
    """Fetch OpenAPI spec from URL and save to file."""
    print(f"Fetching OpenAPI schema from {openapi_url}")
    with urllib.request.urlopen(openapi_url, timeout=30) as response:
        spec = json.loads(response.read().decode("utf-8"))

    normalize_operation_ids(spec)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote normalized schema to {spec_path}")


def write_client_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    package_name: str = DEFAULT_PACKAGE_NAME,
) -> None:
    """Write openapi-python-client configuration file."""
    config_path.write_text(
        "\n".join(
            [
                f"project_name_override: {package_name.replace('_', '-')}",
                f"package_name_override: {package_name}",
                "# Disable post hooks for speed/reproducibility. Run ruff separately if you want.",
                "post_hooks: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {config_path}")


def run_openapi_python_client(
    spec_path: Path = DEFAULT_SPEC_PATH,
    generated_project_dir: Path = DEFAULT_GENERATED_PROJECT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    """Run openapi-python-client to generate typed client."""
    if generated_project_dir.exists():
        shutil.rmtree(generated_project_dir)

    cmd = [
        "openapi-python-client",
        "generate",
        "--path",
        str(spec_path),
        "--output-path",
        str(generated_project_dir),
        "--overwrite",
        "--config",
        str(config_path),
        "--meta",
        "none",
    ]

    print("Generating typed client package with openapi-python-client")
    try:
        generated_project_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "openapi-python-client is not installed. Install it with:\n"
            "  pip install openapi-python-client\n"
            "or:\n"
            "  pipx install openapi-python-client --include-deps"
        ) from exc


def package_dir(generated_project_dir: Path = DEFAULT_GENERATED_PROJECT_DIR) -> Path:
    """Get path to generated client package."""
    if not generated_project_dir.exists():
        raise SystemExit(
            f"Expected generated package at {generated_project_dir}, but it was not found."
        )
    return generated_project_dir


def module_import_path_for_file(
    py_file: Path, generated_project_dir: Path = DEFAULT_GENERATED_PROJECT_DIR
) -> str:
    """Get module import path for a generated file."""
    rel = py_file.relative_to(generated_project_dir).with_suffix("")
    return ".".join(rel.parts)


def find_operation_modules(
    generated_project_dir: Path = DEFAULT_GENERATED_PROJECT_DIR,
) -> list[OperationModule]:
    """Find all operation modules in generated client package."""
    api_dir = package_dir(generated_project_dir) / "api"
    modules: list[OperationModule] = []
    used_names: set[str] = set()

    for py_file in sorted(api_dir.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        has_sync = any(
            isinstance(node, ast.FunctionDef) and node.name == "sync"
            for node in tree.body
        )
        if not has_sync:
            continue

        base_name = snake_case(py_file.stem)
        clean_name = clean_function_name(base_name)
        facade_name = clean_name
        counter = 2
        while facade_name in used_names:
            facade_name = f"{clean_name}_{counter}"
            counter += 1
        used_names.add(facade_name)

        modules.append(
            OperationModule(
                facade_name=facade_name,
                module_import_path=module_import_path_for_file(
                    py_file, generated_project_dir
                ),
                alias=f"_op_{facade_name}",
                path=py_file,
            )
        )

    if not modules:
        raise SystemExit(
            "No generated operation modules with a sync() function were found."
        )

    return modules


class QualifyAnnotationNames(ast.NodeTransformer):
    def __init__(self, alias: str) -> None:
        self.alias = alias

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in SAFE_ANNOTATION_NAMES:
            return node
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=self.alias, ctx=ast.Load()),
                attr=node.id,
                ctx=ast.Load(),
            ),
            node,
        )


def qualify_expr(node: ast.AST | None, alias: str) -> str | None:
    if node is None:
        return None
    cloned = ast.fix_missing_locations(
        QualifyAnnotationNames(alias).visit(ast.fix_missing_locations(node))
    )
    return ast.unparse(cloned)


def get_sync_function(py_file: Path) -> ast.FunctionDef:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync":
            return node
    raise RuntimeError(f"No sync() function found in {py_file}")


def defaults_for_args(
    args: list[ast.arg], defaults: list[ast.expr]
) -> dict[str, ast.expr | None]:
    result: dict[str, ast.expr | None] = {arg.arg: None for arg in args}
    if not defaults:
        return result
    offset = len(args) - len(defaults)
    for index, default in enumerate(defaults):
        result[args[offset + index].arg] = default
    return result


def render_arg(arg: ast.arg, default: ast.expr | None, alias: str) -> str:
    annotation = qualify_expr(arg.annotation, alias) if arg.annotation else "Any"
    rendered = f"{arg.arg}: {annotation}"
    if default is not None:
        rendered += f" = {qualify_expr(default, alias)}"
    return rendered


def render_signature_and_call(
    sync_fn: ast.FunctionDef, alias: str
) -> tuple[str, str, str]:
    pos_args = [*sync_fn.args.posonlyargs, *sync_fn.args.args]
    pos_defaults = defaults_for_args(pos_args, sync_fn.args.defaults)
    kw_defaults = {
        arg.arg: default
        for arg, default in zip(sync_fn.args.kwonlyargs, sync_fn.args.kw_defaults)
    }

    rendered_params: list[str] = []
    call_parts: list[str] = []

    for arg in pos_args:
        if arg.arg == "client":
            continue
        rendered_params.append(render_arg(arg, pos_defaults[arg.arg], alias))
        call_parts.append(f"{arg.arg}={arg.arg}")

    kw_rendered: list[str] = []
    for arg in sync_fn.args.kwonlyargs:
        if arg.arg == "client":
            continue
        kw_rendered.append(render_arg(arg, kw_defaults.get(arg.arg), alias))
        call_parts.append(f"{arg.arg}={arg.arg}")

    if kw_rendered:
        rendered_params.append("*")
        rendered_params.extend(kw_rendered)

    return_annotation = (
        qualify_expr(sync_fn.returns, alias) if sync_fn.returns else "Any"
    )
    return ", ".join(rendered_params), ", ".join(call_parts), return_annotation or "Any"


def extract_class_docstring(py_file: Path, class_name: str) -> str | None:
    """Extract the docstring from a specific class in a Python file."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_docstring(node)
    return None


def extract_sync_docstring(py_file: Path) -> str | None:
    """Extract the docstring from the sync() function in a generated module."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync":
            return ast.get_docstring(node)
    return None


def extract_attributes_from_docstring(docstring: str | None) -> list[str]:
    """Extract attribute descriptions from a class docstring."""
    if not docstring:
        return []

    attributes = []
    in_attributes = False

    for line in docstring.split("\n"):
        line_stripped = line.strip()
        if "Attributes:" in line_stripped:
            in_attributes = True
            continue
        if in_attributes:
            if line_stripped and not line_stripped.startswith("Attributes:"):
                if line_stripped.startswith("Returns:") or line_stripped.startswith(
                    "Raises:"
                ):
                    break
                clean_line = line_stripped.lstrip("-").strip()
                if clean_line:
                    attributes.append(clean_line)
            elif not line_stripped:
                continue

    return attributes


def extract_returns_from_sync_docstring(sync_docstring: str | None) -> str | None:
    """Extract the Returns section from sync function docstring."""
    if not sync_docstring:
        return None

    in_returns = False
    returns_lines = []

    for line in sync_docstring.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("Returns:"):
            in_returns = True
            continue
        if in_returns:
            if line_stripped.startswith(("Args:", "Raises:")):
                break
            if line_stripped:
                returns_lines.append(line_stripped)

    if returns_lines:
        result = " ".join(returns_lines)
        # Clean up type annotations like "str: " at the beginning
        if ": " in result and not result.startswith("    "):
            parts = result.split(": ", 1)
            if len(parts[0].split()) == 1 and parts[0] in (
                "str",
                "int",
                "float",
                "bool",
                "dict",
                "list",
                "tuple",
                "Any",
                "None",
            ):
                return parts[1]
        return result
    return None


def format_operation_docstring(module: OperationModule) -> str:
    """Generate a well-structured docstring for the facade function."""
    sync_docstring = extract_sync_docstring(module.path)
    path_match = re.search(
        r'"url":\s*f?"([^"]+)"', module.path.read_text(encoding="utf-8")
    )
    path = path_match.group(1) if path_match else module.facade_name

    description_lines = []
    if sync_docstring:
        in_description = True
        for line in sync_docstring.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith(("Args:", "Returns:", "Raises:")):
                in_description = False
            if in_description and line_stripped:
                description_lines.append(line_stripped)
    description = " ".join(description_lines)
    description = " ".join(description.split())

    body_param_name = "body"
    form_model_name = None
    response_model_name = None

    tree = ast.parse(module.path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync":
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == "body" and arg.annotation:
                    ann_str = ast.unparse(arg.annotation)
                    form_model_name = ann_str.split(".")[-1].split("|")[0].strip()
                    break

    form_model_attrs = []
    response_model_attrs = []

    models_dir = module.path.parent.parent.parent / "models"
    if form_model_name and models_dir.exists():
        form_model_file = models_dir / f"{snake_case(form_model_name)}.py"
        if form_model_file.exists():
            form_model_doc = extract_class_docstring(form_model_file, form_model_name)
            form_model_attrs = extract_attributes_from_docstring(form_model_doc)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync" and node.returns:
            ret_str = ast.unparse(node.returns)
            if "ResponseModel" in ret_str:
                parts = ret_str.replace(" ", "").split("|")
                for part in parts:
                    if "ResponseModel" in part:
                        response_model_name = part.strip()
                        break

    # Extract result_type from ResponseModel
    result_type = None
    if response_model_name and models_dir.exists():
        response_model_file = models_dir / f"{snake_case(response_model_name)}.py"
        if response_model_file.exists():
            response_model_doc = extract_class_docstring(
                response_model_file, response_model_name
            )
            response_model_attrs = extract_attributes_from_docstring(response_model_doc)
            # Extract result_type from the result attribute
            model_tree = ast.parse(response_model_file.read_text(encoding="utf-8"))
            for node in model_tree.body:
                if isinstance(node, ast.ClassDef) and node.name == response_model_name:
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            if item.target.id == "result" and item.annotation:
                                result_type = ast.unparse(item.annotation)

    doc_lines = []
    if description:
        doc_lines.append(description)

    if form_model_attrs:
        doc_lines.append(f"Args:")
        doc_lines.append(f"    {body_param_name} ({form_model_name}):")
        for attr in form_model_attrs:
            doc_lines.append(f"        - {attr}")

    # Use result_type as ground truth for Returns
    if result_type:
        returns_value = "dict" if result_type == "Any" else result_type
        doc_lines.append("Returns:")
        doc_lines.append(f"    {returns_value}")
    elif response_model_attrs:
        doc_lines.append("Returns:")
        for attr in response_model_attrs:
            doc_lines.append(f"    - {attr}")
    else:
        returns_desc = extract_returns_from_sync_docstring(sync_docstring)
        if returns_desc:
            doc_lines.append(f"Returns:")
            doc_lines.append(f"    {returns_desc}")

    return "\n".join(doc_lines)


def operation_docstring(module: OperationModule) -> str:
    """Generate docstring for the facade function."""
    return format_operation_docstring(module)


def load_openapi_spec(spec_path: Path) -> dict[str, Any]:
    """Load OpenAPI specification from JSON file.

    Args:
        spec_path: Path to the OpenAPI JSON file

    Returns:
        Parsed OpenAPI specification as dictionary
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"OpenAPI spec not found at {spec_path}")

    spec_text = spec_path.read_text(encoding="utf-8")
    return json.loads(spec_text)


def resolve_ref(ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref reference in the OpenAPI spec.

    Args:
        ref: Reference string (e.g., "#/components/schemas/MyModel")
        spec: OpenAPI specification dictionary

    Returns:
        Resolved schema dictionary
    """
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported reference format: {ref}")

    parts = ref[2:].split("/")
    result = spec
    for part in parts:
        if isinstance(result, dict):
            result = result.get(part, {})
        else:
            raise ValueError(f"Cannot resolve ref: {ref}")

    return result


def convert_schema_to_primitives(
    schema: dict[str, Any],
    spec: dict[str, Any],
    visited_refs: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 5,
) -> str:
    """Convert OpenAPI schema to primitive type description.

    Args:
        schema: OpenAPI schema dictionary
        spec: Full OpenAPI specification
        visited_refs: Set of already visited $refs to prevent circular references
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Type description string (e.g., "dict with keys: features (list[dict]), type (str)")
    """
    if visited_refs is None:
        visited_refs = set()

    if depth >= max_depth:
        return "..."

    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in visited_refs:
            return "dict"

        visited_refs.add(ref)
        resolved = resolve_ref(ref, spec)
        return convert_schema_to_primitives(
            resolved, spec, visited_refs.copy(), depth + 1, max_depth
        )

    # Handle allOf, oneOf, anyOf
    for combo_key in ["allOf", "oneOf", "anyOf"]:
        if combo_key in schema:
            schemas = schema[combo_key]
            if isinstance(schemas, list) and len(schemas) > 0:
                # Use first schema for simplicity
                return convert_schema_to_primitives(
                    schemas[0], spec, visited_refs.copy(), depth + 1, max_depth
                )

    schema_type = schema.get("type", "object")

    # Handle basic types
    if schema_type == "string":
        return "str"
    elif schema_type == "integer":
        return "int"
    elif schema_type == "number":
        return "float"
    elif schema_type == "boolean":
        return "bool"
    elif schema_type == "null":
        return "None"

    # Handle array
    if schema_type == "array":
        items = schema.get("items", {})
        if items:
            converted_items = convert_schema_to_primitives(
                items, spec, visited_refs.copy(), depth + 1, max_depth
            )
            return f"list[{converted_items}]"
        return "list[Any]"

    # Handle object
    if schema_type == "object":
        properties = schema.get("properties", {})
        if properties:
            attr_descriptions = []
            for prop_name, prop_schema in properties.items():
                converted_type = convert_schema_to_primitives(
                    prop_schema, spec, visited_refs.copy(), depth + 1, max_depth
                )
                attr_descriptions.append(f"{prop_name} ({converted_type})")

            if attr_descriptions:
                return f"dict with keys: {', '.join(attr_descriptions)}"

        return "dict"

    # Fallback
    return "Any"


def get_response_schema_for_operation(
    operation_id: str, spec: dict[str, Any]
) -> dict[str, Any] | None:
    """Get the response schema for an operation from the OpenAPI spec.

    Args:
        operation_id: The operationId to find
        spec: OpenAPI specification dictionary

    Returns:
        Response schema dictionary, or None if not found
    """
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            if operation.get("operationId") == operation_id:
                # Get 200 response schema
                responses = operation.get("responses", {})
                response_200 = responses.get("200", {})
                content = response_200.get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema")
                if schema:
                    return schema
    return None


def format_returns_from_openapi(
    operation_id: str, spec: dict[str, Any], max_depth: int = 10
) -> str:
    """Format Returns docstring section from OpenAPI schema.

    Args:
        operation_id: The operationId to find
        spec: OpenAPI specification dictionary
        max_depth: Maximum recursion depth for type conversion

    Returns:
        Formatted Returns docstring content
    """
    schema = get_response_schema_for_operation(operation_id, spec)
    if not schema:
        return "dict[str, Any]: Response data"

    converted = convert_schema_to_primitives(schema, spec, max_depth=max_depth)

    # Format nicely
    if converted.startswith("dict with keys:"):
        return f"dict[str, Any]: {converted}"
    elif converted in ("dict", "Any"):
        return "dict[str, Any]: Response data"
    else:
        return f"dict[str, Any]: {converted}"


def extract_form_model_info(
    module: OperationModule,
) -> tuple[str | None, list[tuple[str, str, str | None]]]:
    """Extract FormModel class name and its attributes with types and defaults.

    Returns:
        Tuple of (form_model_name, [(attr_name, attr_type, default_value), ...])
    """
    tree = ast.parse(module.path.read_text(encoding="utf-8"))
    form_model_name = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync":
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == "body" and arg.annotation:
                    ann_str = ast.unparse(arg.annotation)
                    form_model_name = ann_str.split(".")[-1].split("|")[0].strip()
                    break

    if not form_model_name:
        return None, []

    models_dir = module.path.parent.parent.parent / "models"
    form_model_file = models_dir / f"{snake_case(form_model_name)}.py"

    if not form_model_file.exists():
        return form_model_name, []

    model_tree = ast.parse(form_model_file.read_text(encoding="utf-8"))
    attributes = []

    for node in model_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == form_model_name:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    attr_name = item.target.id
                    if attr_name == "additional_properties":
                        continue
                    attr_type = (
                        ast.unparse(item.annotation) if item.annotation else "Any"
                    )
                    default_value = None
                    if item.value:
                        default_str = ast.unparse(item.value)
                        if default_str != "UNSET":
                            default_value = default_str
                    attributes.append((attr_name, attr_type, default_value))
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attr_name = target.id
                            if attr_name == "additional_properties":
                                continue
                            attr_type = "Any"
                            default_value = (
                                ast.unparse(item.value) if item.value else None
                            )
                            if default_value == "UNSET":
                                default_value = None
                            attributes.append((attr_name, attr_type, default_value))

    return form_model_name, attributes


def extract_response_model_info(
    module: OperationModule,
) -> tuple[str | None, list[str], str | None]:
    """Extract ResponseModel class name and its attributes.

    Returns:
        Tuple of (response_model_name, [attr_doc_lines], result_type)
    """
    tree = ast.parse(module.path.read_text(encoding="utf-8"))
    response_model_name = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync" and node.returns:
            ret_str = ast.unparse(node.returns)
            if "ResponseModel" in ret_str:
                parts = ret_str.replace(" ", "").split("|")
                for part in parts:
                    if "ResponseModel" in part:
                        response_model_name = part.strip()
                        break

    if not response_model_name:
        return None, [], None

    models_dir = module.path.parent.parent.parent / "models"
    response_model_file = models_dir / f"{snake_case(response_model_name)}.py"

    if not response_model_file.exists():
        return response_model_name, [], None

    model_tree = ast.parse(response_model_file.read_text(encoding="utf-8"))
    attr_docs = []
    result_type = None

    for node in model_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == response_model_name:
            docstring = ast.get_docstring(node)
            if docstring:
                for line in docstring.split("\n"):
                    line_stripped = line.strip().lstrip("-").strip()
                    if line_stripped and "Attributes:" not in line_stripped:
                        attr_docs.append(line_stripped)

            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    if item.target.id == "result" and item.annotation:
                        result_type = ast.unparse(item.annotation)

    return response_model_name, attr_docs, result_type


def extract_operation_id_from_module(module_path: Path) -> str | None:
    """Extract operationId from a generated module file.

    The operationId is derived from the module filename since openapi-python-client
    doesn't store it as a variable.

    Args:
        module_path: Path to the generated module file

    Returns:
        operationId string or None if not found
    """
    # Extract from filename: tool_avenue_of_approaches_post.py -> tool_avenue_of_approaches_post
    operation_id = module_path.stem
    if operation_id and operation_id != "__init__":
        return operation_id
    return None


def generate_wrapper(
    module: OperationModule,
    usage_postfix: str | None = None,
    openapi_spec: dict[str, Any] | None = None,
) -> str:
    """Generate wrapper function with flattened FormModel parameters.

    Args:
        module: Operation module information
        usage_postfix: Optional usage instructions
        openapi_spec: Optional OpenAPI specification for Returns docstring
    """
    form_model_name, form_model_attrs = extract_form_model_info(module)
    response_model_name, response_attr_docs, result_type = extract_response_model_info(
        module
    )

    if not form_model_name or not form_model_attrs:
        sync_fn = get_sync_function(module.path)
        signature, call_args, return_annotation = render_signature_and_call(
            sync_fn, module.alias
        )
        # Always use dict[str, Any] for return type
        return_annotation = "dict[str, Any]"
        doc = operation_docstring(module).replace('"""', "'''")
        if usage_postfix:
            doc = f"{doc}\n\n{usage_postfix}".replace('"""', "'''")
        args = ["client=_get_client()"]
        if call_args:
            args.append(call_args)
        call = ", ".join(args)
        return f'''\ndef {module.facade_name}({signature}) -> {return_annotation}:\n    """{doc}"""\n    return {module.alias}.sync({call})\n'''

    params = []
    for attr_name, attr_type, default_value in form_model_attrs:
        qualified_type = attr_type
        if attr_type not in SAFE_ANNOTATION_NAMES and not any(
            builtin in attr_type
            for builtin in ["list", "dict", "tuple", "set", "frozenset"]
        ):
            if "Model" in attr_type:
                qualified_type = f"{module.alias}.{attr_type}"
        if "Unset" in qualified_type:
            parts = qualified_type.split("|")
            parts = [p.strip() for p in parts if "Unset" not in p]
            qualified_type = " | ".join(parts) if parts else "Any"
        if default_value is not None:
            params.append(f"{attr_name}: {qualified_type} = {default_value}")
        else:
            params.append(f"{attr_name}: {qualified_type}")

    signature = ", ".join(params)

    kwargs_parts = []
    for attr_name, _, _ in form_model_attrs:
        kwargs_parts.append(f"{attr_name}={attr_name}")
    kwargs_str = ", ".join(kwargs_parts)

    # Always use dict[str, Any] for return type annotation
    return_annotation = "dict[str, Any]"

    doc_lines = []
    sync_docstring = extract_sync_docstring(module.path)
    if sync_docstring:
        description_lines = []
        in_description = True
        for line in sync_docstring.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith(("Args:", "Returns:", "Raises:")):
                in_description = False
            if in_description and line_stripped:
                description_lines.append(line_stripped)
        description = " ".join(description_lines)
        description = " ".join(description.split())
        if description:
            doc_lines.append(description)

    doc_lines.append("Args:")
    models_dir = module.path.parent.parent.parent / "models"
    form_model_file = models_dir / f"{snake_case(form_model_name)}.py"
    form_model_doc = (
        extract_class_docstring(form_model_file, form_model_name)
        if form_model_file.exists()
        else None
    )

    for attr_name, attr_type, default_value in form_model_attrs:
        attr_doc_line = ""
        if form_model_doc:
            for line in form_model_doc.split("\n"):
                line_stripped = line.strip().lstrip("-").strip()
                if line_stripped.startswith(f"{attr_name} ") and ":" in line_stripped:
                    colon_idx = line_stripped.index(":")
                    attr_doc_line = line_stripped[colon_idx + 1 :].strip()
                    attr_doc_line = (
                        attr_doc_line.replace(" | Unset", "")
                        .replace("| Unset", "")
                        .strip()
                    )
                    if default_value is not None and "Default:" not in attr_doc_line:
                        attr_doc_line += f" Default: {default_value}."
                    break
        clean_type = attr_type.replace(" | Unset", "").replace("| Unset", "").strip()
        if attr_doc_line:
            doc_lines.append(f"    {attr_name} ({clean_type}): {attr_doc_line}")
        else:
            doc_lines.append(f"    {attr_name} ({clean_type})")

    # Use OpenAPI schema for Returns docstring if available
    returns_from_openapi = False
    if openapi_spec:
        operation_id = extract_operation_id_from_module(module.path)
        if operation_id:
            returns_section = format_returns_from_openapi(
                operation_id, openapi_spec, max_depth=10
            )
            doc_lines.append("Returns:")
            doc_lines.append(f"    {returns_section}")
            returns_from_openapi = True

    # Fallback to old method if OpenAPI didn't work
    if not returns_from_openapi:
        if result_type:
            returns_value = "dict" if result_type == "Any" else result_type
            doc_lines.append("Returns:")
            doc_lines.append(f"    {returns_value}")
        elif response_attr_docs:
            doc_lines.append("Returns:")
            for attr_doc in response_attr_docs:
                if attr_doc and "Attributes:" not in attr_doc:
                    clean_attr_doc = attr_doc.replace(" | Unset", "").replace(
                        "| Unset", ""
                    )
                    doc_lines.append(f"    {clean_attr_doc}")

    doc = "\n".join(doc_lines).replace('"""', "'''")
    if usage_postfix:
        doc = f"{doc}\n\n{usage_postfix}".replace('"""', "'''")

    body_lines = [
        f"@code",
        f"def {module.facade_name}({signature}) -> {return_annotation}:",
        f'    """{doc}"""',
        f"    body = {module.alias}.{form_model_name}({kwargs_str})",
        f"    response = {module.alias}.sync(client=_get_client(), body=body)",
        f"    return response",
    ]

    return "\n".join(body_lines) + "\n"


def generate_test_main(modules: list[OperationModule]) -> str:
    """Generate __main__ block with test calls for each wrapper function."""
    test_calls = []

    for module in modules:
        form_model_name, form_model_attrs = extract_form_model_info(module)

        if not form_model_attrs:
            continue

        # Build mock arguments based on parameter types
        mock_args = []
        for attr_name, attr_type, default_value in form_model_attrs:
            if default_value is not None:
                # Use default value if available
                mock_args.append(f"{attr_name}={default_value}")
            elif "list" in attr_type.lower():
                # Mock list with sample coordinates (triangle polygon)
                mock_args.append(
                    f"{attr_name}=[[52.15995472769272, 11.175482626854976], [52.12357669105406, 11.071887076720898], [52.099033902796656, 11.234383588578686]]"
                )
            elif "str" in attr_type.lower():
                # Mock string
                mock_args.append(f'{attr_name}="test"')
            elif "int" in attr_type.lower():
                # Mock int
                mock_args.append(f"{attr_name}=1")
            elif "float" in attr_type.lower():
                # Mock float
                mock_args.append(f"{attr_name}=1.0")
            else:
                # Generic mock
                mock_args.append(f"{attr_name}=None")

        args_str = ", ".join(mock_args)
        test_calls.append(f'    print("Testing {module.facade_name}...")')
        test_calls.append(f"    try:")
        test_calls.append(f"        result = {module.facade_name}({args_str})")
        test_calls.append(
            f'        print(f"  Success! Result type: {{type(result).__name__}}")'
        )
        test_calls.append(f"    except Exception as e:")
        test_calls.append(f'        print(f"  Error: {{e}}")')
        test_calls.append(f"")

    if not test_calls:
        return ""

    main_block = '''

if __name__ == "__main__":
    """Test all wrapper functions with mock data."""
    print("Testing MCP-O wrapper functions...")
    print("=" * 60)
    print()
    
'''
    main_block += "\n".join(test_calls)
    main_block += """
    print("=" * 60)
    print("All tests completed.")
"""

    return main_block


def _write_reexport_wrapper(
    modules: list[OperationModule], root_facade_path: Path, package_name: str
) -> None:
    """Generate re-export wrapper in function_library root.

    This allows agents to import functions directly from {package_name}_imports
    while the actual implementation stays in function_library.openapi.{package_name}.{package_name}_functions
    where the {package_name} client imports work correctly.
    """
    function_names = [module.facade_name for module in modules]

    imports = "\n".join(f"    {name}," for name in function_names)

    all_list = "\n".join(f"    '{name}'," for name in function_names)

    content = f"""# Auto-generated by generate_mcpo_client_and_facade.py.
# Do not edit by hand; regenerate instead.
# 
# This is a re-export wrapper that imports from the nested location.
# The actual implementation is in function_library.openapi.{package_name}.{package_name}_functions
# where the {package_name} imports work correctly.

# Import with fallback for both package and direct import
try:
    from .openapi.{package_name}.{package_name}_functions import (
{imports}
    )
except ImportError:
    from openapi.{package_name}.{package_name}_functions import (
{imports}
    )

__all__ = [
{all_list}
]
"""

    root_facade_path.parent.mkdir(parents=True, exist_ok=True)
    root_facade_path.write_text(content, encoding="utf-8")
    print(
        f"Wrote re-export wrapper to {root_facade_path} with {len(modules)} functions"
    )


def generate_mcpo_client(
    openapi_url: str,
    generated_projects_dir: str,
    package_name: str = DEFAULT_PACKAGE_NAME,
    api_key: str | None = None,
    usage_postfix: str | None = None,
    skip_sync_check: bool = False,
) -> tuple[Path, Path]:
    """
    Generate typed OpenAPI client and facade wrappers.

    Args:
        openapi_url: URL to OpenAPI JSON schema (base URL will be derived from this)
        generated_projects_dir: Directory to generate client code into
        package_name: Name for the generated Python package
        api_key: Optional API key for authentication
        usage_postfix: Optional usage instructions appended to wrapper docstrings
        skip_sync_check: If True, skip sync check and always regenerate

    Returns:
        Tuple of (generated_client_dir, facade_file_path)
    """
    generated_projects_dir_path = Path(generated_projects_dir)
    generated_project_dir = generated_projects_dir_path / package_name
    spec_path = generated_projects_dir_path / "openapi.json"
    config_path = generated_projects_dir_path / "openapi-python-client-config.yaml"
    # Use package_name-based naming: geo_mcp -> geo_mcp_functions.py
    facade_path = generated_projects_dir_path / f"{package_name}_functions.py"

    # Delete old generated code before regenerating
    if generated_project_dir.exists():
        import shutil

        print(f"Removing old generated code: {generated_project_dir}")
        shutil.rmtree(generated_project_dir)
    if facade_path.exists():
        facade_path.unlink()
        print(f"Removing old facade file: {facade_path}")
    if spec_path.exists():
        spec_path.unlink()
    if config_path.exists():
        config_path.unlink()

    # Derive base_url from openapi_url (remove path like /openapi.json)
    from urllib.parse import urlparse

    parsed = urlparse(openapi_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # Fetch and prepare spec
    fetch_and_prepare_spec(openapi_url, spec_path)

    # Write client config
    write_client_config(config_path, package_name)

    # Generate typed client
    run_openapi_python_client(spec_path, generated_project_dir, config_path)

    # Find operation modules
    modules = find_operation_modules(generated_project_dir)

    # Generate facade with derived base_url
    _write_facade_custom(
        modules,
        facade_path,
        generated_project_dir,
        base_url,
        api_key,
        usage_postfix,
        package_name,
    )

    print(
        f"Done. Import from {package_name}_functions.py, not directly from the generated client."
    )

    # Generate re-export wrapper in function_library root for easy agent import
    # Use package_name-based naming: geo_mcp -> geo_mcp_imports.py
    function_library_root = Path(__file__).parent.parent / "function_library"
    root_facade_path = function_library_root / f"{package_name}_imports.py"
    _write_reexport_wrapper(modules, root_facade_path, package_name)

    return generated_project_dir, facade_path


def _write_facade_custom(
    modules: list[OperationModule],
    facade_path: Path,
    generated_project_dir: Path,
    base_url: str = "http://localhost:8000",
    api_key: str | None = None,
    usage_postfix: str | None = None,
    package_name: str = "mcpo_client",
) -> None:
    """Generate facade file with custom base_url."""
    # Load OpenAPI spec for Returns docstrings
    # Spec is saved in parent directory (generated_projects_dir / "openapi.json")
    spec_path = generated_project_dir.parent / "openapi.json"
    openapi_spec = None
    if spec_path.exists():
        try:
            openapi_spec = load_openapi_spec(spec_path)
            print(
                f"✓ Loaded OpenAPI spec from {spec_path} ({len(openapi_spec.get('paths', {}))} paths)"
            )
        except Exception as e:
            print(f"✗ Warning: Could not load OpenAPI spec: {e}")
    else:
        print(f"✗ Warning: OpenAPI spec not found at {spec_path}")

    imports = "\n".join(
        f"from {module.module_import_path} import sync as _unused_{module.facade_name}_sync  # noqa: F401"
        for module in []
    )
    operation_imports = "\n".join(
        f"from {package_name}.{module.module_import_path.rsplit('.', 1)[0]} import {module.module_import_path.rsplit('.', 1)[1]} as {module.alias}"
        for module in modules
    )
    wrappers = "\n".join(
        generate_wrapper(module, usage_postfix=usage_postfix, openapi_spec=openapi_spec)
        for module in modules
    )
    test_main = generate_test_main(modules)

    api_key_default = f'"{api_key}"' if api_key else "None"

    content = f'''# Auto-generated by generate_mcpo_client_and_facade.py.\n# Do not edit by hand; regenerate instead.\n\nfrom __future__ import annotations\n\nimport os\nimport sys\nfrom functools import lru_cache\nfrom pathlib import Path\nfrom typing import Any\n\n# Import code decorator - works both as package and direct import\ntry:\n    from ... import code  # When in function_library/openapi/{package_name}/\nexcept ImportError:\n    # Direct import (when function_library is in sys.path)\n    sys.path.insert(0, str(Path(__file__).parent.parent.parent))\n    from __init__ import code\n\n_GENERATED_PROJECT = Path(__file__).resolve().parent / "{package_name}"\n_parent_dir = str(_GENERATED_PROJECT.parent)\nif _parent_dir not in sys.path:\n    sys.path.insert(0, _parent_dir)\n\nfrom {package_name} import AuthenticatedClient, Client\n{operation_imports}\n\n\n@lru_cache(maxsize=1)\ndef _get_client() -> Client | AuthenticatedClient:\n    base_url = os.getenv("MCPO_BASE_URL", "{base_url}").rstrip("/")\n    api_key = os.getenv("MCPO_API_KEY", {api_key_default})\n\n    if api_key:\n        return AuthenticatedClient(base_url=base_url, token=api_key)\n\n    return Client(base_url=base_url)\n\n\ndef clear_client_cache() -> None:\n    """Clear cached client, useful if MCPO_BASE_URL or MCPO_API_KEY changes at runtime."""\n    _get_client.cache_clear()\n\n{wrappers}\n{test_main}\n'''

    content = content.replace(imports, "")
    facade_path.parent.mkdir(parents=True, exist_ok=True)
    facade_path.write_text(content, encoding="utf-8")
    print(f"Wrote {facade_path} with {len(modules)} endpoint functions")


def ensure_mcpo_functions_synced() -> None:
    """
    Ensure all MCPO-generated functions are in sync with their OpenAPI specs.

    This function fetches the current OpenAPI specs from their remote URLs,
    compares them with saved versions, and regenerates the client code if
    they differ.

    Raises:
        Exception: If spec fetch or client generation fails
    """
    openapi_dir = BASE_DIR

    for server in OPENAPI_SERVERS:
        package_name = server["package_name"]
        openapi_url = server["openapi_url"]
        api_key = server.get("api_key")
        usage_postfix = server.get("usage_postfix")

        package_dir = openapi_dir / package_name
        if not package_dir.exists():
            print(f"Package directory not found for {package_name}, generating...")
            project_dir = BASE_DIR / server["package_name"]
            project_dir.mkdir(exist_ok=True, parents=True)
            generate_mcpo_client(
                openapi_url=server["openapi_url"],
                generated_projects_dir=str(project_dir),
                package_name=server["package_name"],
                api_key=server.get("api_key"),
                usage_postfix=server.get("usage_postfix"),
            )
            continue

        print(f"Checking sync for {package_name}...")
        try:
            generated_dir, facade_path, was_regenerated = check_and_generate_if_needed(
                openapi_url=openapi_url,
                generated_projects_dir=package_dir,
                package_name=package_name,
                api_key=api_key,
                usage_postfix=usage_postfix,
            )

            if was_regenerated:
                print(f"Regenerated {package_name} client")
            else:
                print(f"{package_name} client is up to date")

        except Exception as e:
            raise RuntimeError(
                f"Failed to sync MCPO functions for {package_name}: {e}"
            ) from e


if __name__ == "__main__":
    ensure_mcpo_functions_synced()
