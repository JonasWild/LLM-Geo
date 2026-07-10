"""Hard-coded, reviewed OpenAPI services allowed to generate trusted operations."""

from __future__ import annotations

from typing import Any


OPENAPI_SERVERS: list[dict[str, Any]] = [
    {
        "service": "geo_mcp",
        # Use exactly one schema source: openapi_url or openapi_path. Relative paths
        # are resolved from the repository root.
        "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194/openapi.json",
        # "openapi_path": "openapi/geo_mcp.json",
        # This is the actual endpoint used by generated operations. It is independent
        # of the URL or file from which the OpenAPI document was loaded.
        "base_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194",
        "api_key_environment": "LLM_GEO_OPENAPI_GEO_MCP_API_KEY",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
]
