"""Hard-coded, reviewed OpenAPI services allowed to generate trusted operations."""

from __future__ import annotations

import os
from typing import Any
import dotenv

dotenv.load_dotenv(override=True)

OPENAPI_SERVERS: list[dict[str, Any]] = [
    # {
    #     "service": "geo_mcp",
    #     # Use exactly one schema source: openapi_url or openapi_path. Relative paths
    #     # are resolved from the repository root.
    #     "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194/openapi.json",
    #     # "openapi_path": "openapi/geo_mcp.json",
    #     # This is the actual endpoint used by generated operations. It is independent
    #     # of the URL or file from which the OpenAPI document was loaded.
    #     "base_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194",
    #     "api_key_environment": "LLM_GEO_OPENAPI_GEO_MCP_API_KEY",
    #     "auth_header": "Authorization",
    #     "auth_scheme": "Bearer",
    # },
    {
        "service": "geo_mcp",
        "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194/openapi.json",
        "base_url": "http://zeu-ki-02.zeu.de.airbusds.corp:31194",
        "api_key_environment": "LLM_GEO_OPENAPI_GEO_MCP_API_KEY",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
    {
        "service": "SitaWare Interpreter API",
        "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:32203/openapi.json",
        "base_url": "http://zeu-ki-02.zeu.de.airbusds.corp:32203",
        "api_key_environment": "LLM_GEO_OPENAPI_GEO_MCP_API_KEY",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
    {
        "service": "MIL-STD-2525 Symbol Generator",
        "openapi_url": "http://zeu-ki-02.zeu.de.airbusds.corp:32102/openapi.json",
        "base_url": "http://zeu-ki-02.zeu.de.airbusds.corp:32102",
        "api_key_environment": "LLM_GEO_OPENAPI_GEO_MCP_API_KEY",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
    {
        "service": "SitaWare API",
        "openapi_path": r"C:\Users\WIJO188\Projects\LLM-Geo\llm_geo\operations\openapi\data\sitaware_openapi.json",
        "base_url": os.environ.get("SITAWARE_BASE_URL"),
        "api_key_environment": os.environ.get("SITAWARE_TOKEN"),
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
]
