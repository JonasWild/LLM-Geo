#!/usr/bin/env python3
"""Synchronize hard-coded OpenAPI servers with generated ``@code`` operations.

The historical filename is retained as a familiar development entry point. Unlike
the previous implementation, generation reads OpenAPI directly and does not create
or reverse-engineer an ``openapi-python-client`` package.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from llm_geo.operations.openapi.generator import (
    DEFAULT_OUTPUT_DIRECTORY,
    generate_openapi_operations,
)
from llm_geo.operations.openapi.servers import OPENAPI_SERVERS

PROJECT_ROOT = Path(__file__).parents[2]


def fetch_openapi_spec(
    openapi_url: str, *, api_key_environment: str | None = None,
    auth_header: str = "Authorization", auth_scheme: str = "Bearer"
) -> dict[str, Any]:
    """Download and decode one OpenAPI JSON document."""
    headers: dict[str, str] = {}
    api_key = os.getenv(api_key_environment) if api_key_environment else None
    if api_key:
        headers[auth_header] = f"{auth_scheme} {api_key}".strip()
    request = urllib.request.Request(openapi_url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"OpenAPI document at {openapi_url} is not a JSON object")
    return value


def load_openapi_spec(server: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Load one configured schema from exactly one URL or local JSON file."""
    openapi_url = server.get("openapi_url")
    openapi_path = server.get("openapi_path")
    if bool(openapi_url) == bool(openapi_path):
        raise ValueError(
            f"{server.get('service', 'OpenAPI service')} must configure exactly one "
            "of openapi_url or openapi_path"
        )
    if openapi_url:
        source = str(openapi_url)
        return (
            fetch_openapi_spec(
                source,
                api_key_environment=server.get("api_key_environment"),
                auth_header=str(server.get("auth_header", "Authorization")),
                auth_scheme=str(server.get("auth_scheme", "Bearer")),
            ),
            source,
        )
    path = Path(str(openapi_path)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"OpenAPI document does not exist: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"OpenAPI document at {path} is not a JSON object")
    return value, str(path)


def ensure_openapi_operations_synced(
    *, output_directory: Path = DEFAULT_OUTPUT_DIRECTORY, force: bool = False
) -> None:
    """Fetch every configured server and generate validated operation modules."""
    for server in OPENAPI_SERVERS:
        service = str(server["service"])
        api_key_environment = server.get("api_key_environment")
        auth_header = str(server.get("auth_header", "Authorization"))
        auth_scheme = str(server.get("auth_scheme", "Bearer"))
        base_url = str(server.get("base_url", "")).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"{service} must configure base_url as an absolute HTTP(S) URL"
            )
        spec, source = load_openapi_spec(server)
        result = generate_openapi_operations(
            spec,
            service=service,
            default_base_url=base_url,
            output_directory=output_directory,
            source=source,
            api_key_environment=api_key_environment,
            auth_header=auth_header,
            auth_scheme=auth_scheme,
            force=force,
        )
        state = "generated" if result.changed else "up to date"
        print(f"{service}: {state} ({len(result.operation_ids)} operations)")


def ensure_mcpo_functions_synced() -> None:
    """Backward-compatible alias for the former synchronization function."""
    ensure_openapi_operations_synced()


if __name__ == "__main__":
    ensure_openapi_operations_synced()
