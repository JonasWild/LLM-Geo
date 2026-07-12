"""Render normalized OpenAPI definitions as strict ``@code`` functions."""

from __future__ import annotations

from llm_geo.operations.openapi.models import OpenAPIDefinition, ParameterDefinition


def _default(parameter: ParameterDefinition) -> str:
    if parameter.has_default:
        return f" = {parameter.default!r}"
    if not parameter.required:
        return " = None"
    return ""


def _mapping_lines(
    variable: str, parameters: list[ParameterDefinition]
) -> list[str]:
    required = [parameter for parameter in parameters if parameter.required]
    optional = [parameter for parameter in parameters if not parameter.required]
    entries = ", ".join(
        f"{parameter.wire_name!r}: {parameter.python_name}" for parameter in required
    )
    lines = [f"    {variable}: dict[str, object] = {{{entries}}}"]
    for parameter in optional:
        lines.extend(
            [
                f"    if {parameter.python_name} is not None:",
                f"        {variable}[{parameter.wire_name!r}] = {parameter.python_name}",
            ]
        )
    return lines


def render_operation(
    definition: OpenAPIDefinition,
    *,
    service: str,
    default_base_url: str,
    api_key_environment: str | None,
    auth_header: str,
    auth_scheme: str,
    kind: str,
    returns_geojson: bool,
) -> str:
    """Render one definition as a module-level decorated function."""
    signature = ",\n".join(
        f"    {parameter.python_name}: {parameter.annotation}{_default(parameter)}"
        for parameter in definition.parameters
    )
    if signature:
        signature = f"\n{signature},\n"
    doc_lines = [f'    """{definition.description}', "", "    Args:"]
    if definition.parameters:
        doc_lines.extend(
            f"        {parameter.python_name}: {parameter.description}"
            for parameter in definition.parameters
        )
    return_annotation = definition.return_annotation
    return_description = definition.return_description
    if returns_geojson:
        return_annotation = "GeoDataFrame"
        return_description = f'GeoDataFrame with provenance metadata in `.attrs["provenance"]` ({return_description})'
    doc_lines.extend(
        ["", "    Returns:", f"        {return_description}", '    """']
    )
    body: list[str] = []
    grouped = {
        location: [
            parameter
            for parameter in definition.parameters
            if parameter.location == location
        ]
        for location in ("path", "query", "header", "body")
    }
    for location, variable in (
        ("path", "path_parameters"),
        ("query", "query_parameters"),
        ("header", "header_parameters"),
    ):
        if grouped[location]:
            body.extend(_mapping_lines(variable, grouped[location]))
    body_parameters = grouped["body"]
    body_argument = "None"
    if body_parameters:
        if len(body_parameters) == 1 and body_parameters[0].wire_name == "body":
            body_argument = body_parameters[0].python_name
        else:
            body.extend(_mapping_lines("json_body", body_parameters))
            body_argument = "json_body"
    call_lines = [
        "    payload = invoke_json(" if returns_geojson else "    return invoke_json(",
        f"        service={service!r},",
        f"        method={definition.method!r},",
        f"        path={definition.path!r},",
        f"        default_base_url={default_base_url!r},",
    ]
    for location, variable in (
        ("path", "path_parameters"),
        ("query", "query_parameters"),
        ("header", "header_parameters"),
    ):
        if grouped[location]:
            call_lines.append(f"        {variable}={variable},")
    if body_parameters:
        call_lines.append(f"        json_body={body_argument},")
    if api_key_environment:
        call_lines.append(f"        api_key_environment={api_key_environment!r},")
    call_lines.extend(
        [
            f"        auth_header={auth_header!r},",
            f"        auth_scheme={auth_scheme!r},",
            "    )",
        ]
    )
    if returns_geojson:
        source = f"{service} {definition.method} {definition.path}"
        call_lines.append(f"    return geojson_to_geodataframe(payload, source={source!r})")
    rendered_body = "\n".join(body + call_lines)
    return "\n".join(
        [
            f"@code(kind={kind!r})",
            f"def {definition.function_name}({signature}) -> {return_annotation}:",
            *doc_lines,
            rendered_body,
        ]
    )


def render_module(
    definitions: tuple[OpenAPIDefinition, ...],
    *,
    service: str,
    default_base_url: str,
    kinds: dict[str, str],
    returns_geojson: dict[str, bool] | None = None,
    api_key_environment: str | None = None,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
) -> str:
    """Render a complete importable generated operations module.

    `kinds` and `returns_geojson` map each definition's `function_name` to its classified
    `@code` kind and whether its response should be converted to a GeoDataFrame (see
    `llm_geo.operations.openapi.classify.classify_operations`).
    """
    returns_geojson = returns_geojson or {}
    wrappers = "\n\n\n".join(
        render_operation(
            definition,
            service=service,
            default_base_url=default_base_url,
            api_key_environment=api_key_environment,
            auth_header=auth_header,
            auth_scheme=auth_scheme,
            kind=kinds[definition.function_name],
            returns_geojson=returns_geojson.get(definition.function_name, False),
        )
        for definition in definitions
    )
    needs_geo_imports = any(
        returns_geojson.get(definition.function_name, False) for definition in definitions
    )
    geo_imports = (
        "from geopandas import GeoDataFrame\n"
        "from llm_geo.operations.openapi.runtime import geojson_to_geodataframe\n"
        if needs_geo_imports else ""
    )
    return (
        '"""Generated trusted operations. Do not edit by hand."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        f"{geo_imports}"
        "from llm_geo.operations import code\n"
        "from llm_geo.operations.openapi.runtime import invoke_json\n\n\n"
        f"{wrappers}\n"
    )
