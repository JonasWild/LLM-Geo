"""Explicit allowlist of generated trusted-operation modules."""

from __future__ import annotations

from importlib import import_module, util

from llm_geo.operations.openapi.parser import python_identifier
from llm_geo.operations.openapi.servers import OPENAPI_SERVERS

# This is an explicit allowlist backed by the reviewed hard-coded server configuration.
# Missing modules are allowed so a fresh checkout works before its first generation.
GENERATED_MODULES = tuple(
    python_identifier(str(server["service"]), "service") for server in OPENAPI_SERVERS
)

for _module_name in GENERATED_MODULES:
    _qualified_name = f"{__name__}.{_module_name}"
    if util.find_spec(_qualified_name) is not None:
        import_module(_qualified_name)

__all__ = ["GENERATED_MODULES"]
