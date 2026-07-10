"""Hard-coded, reviewed OpenAPI services allowed to generate trusted operations."""

from __future__ import annotations

from typing import Any


OPENAPI_SERVERS: list[dict[str, Any]] = [
    {
        "service": "geo_mcp",
        "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194/openapi.json",
        "api_key_environment": "LLM_GEO_OPENAPI_GEO_MCP_API_KEY",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
]
