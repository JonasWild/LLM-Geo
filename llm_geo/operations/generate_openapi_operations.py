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
from urllib.parse import urlsplit

from llm_geo.operations.openapi.generator import (
    DEFAULT_OUTPUT_DIRECTORY,
    generate_openapi_operations,
)
from llm_geo.operations.openapi.servers import OPENAPI_SERVERS


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


def _default_base_url(openapi_url: str, spec: dict[str, Any]) -> str:
    servers = spec.get("servers", [])
    if isinstance(servers, list) and servers and isinstance(servers[0], dict):
        configured = servers[0].get("url")
        if isinstance(configured, str) and configured.startswith(("http://", "https://")):
            return configured.rstrip("/")
    parsed = urlsplit(openapi_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def ensure_openapi_operations_synced(
    *, output_directory: Path = DEFAULT_OUTPUT_DIRECTORY, force: bool = False
) -> None:
    """Fetch every configured server and generate validated operation modules."""
    for server in OPENAPI_SERVERS:
        service = str(server["service"])
        openapi_url = str(server["openapi_url"])
        api_key_environment = server.get("api_key_environment")
        auth_header = str(server.get("auth_header", "Authorization"))
        auth_scheme = str(server.get("auth_scheme", "Bearer"))
        spec = fetch_openapi_spec(
            openapi_url,
            api_key_environment=api_key_environment,
            auth_header=auth_header,
            auth_scheme=auth_scheme,
        )
        result = generate_openapi_operations(
            spec,
            service=service,
            default_base_url=_default_base_url(openapi_url, spec),
            output_directory=output_directory,
            source=openapi_url,
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
