"""LLM-GEO task configuration and executable entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from llm_geo.middleware.logging import configure_logging, get_logger
from llm_geo.operations import registered_operations
from llm_geo.subagents.supervisor import create_geo_agent, run_geo_agent
from llm_geo.system import run_llm_geo
from llm_geo.tools.public_data_providers import PUBLIC_RETRIEVAL_TOOLS

if sys.platform == 'win32':
    import pip_system_certs.wrapt_requests

    pip_system_certs.wrapt_requests.inject_truststore()

def _environment_bool(name: str, default: bool) -> bool:
    """Read a conventional boolean value from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, yes/no, on/off, or 1/0")


def _environment_positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _environment_log_level(name: str, default: int) -> int:
    """Read a Python logging level name from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    level = logging.getLevelNamesMapping().get(raw_value.strip().upper())
    if level is None:
        raise ValueError(f"{name} must be a valid logging level such as INFO or DEBUG")
    return level


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK = "Finde die urbane Gebiete mittels overpass zwischen Magedburg und Hohes Holz und stelle sie als png dar."

TASK = """
```
coords_latlon=[
    (52.15995472769272, 11.175482626854976),
    (52.12357669105406, 11.071887076720898),
    (52.099033902796656, 11.234383588578686),
    (52.093345131708986, 11.291868995068482),
    (52.09865453410721, 11.259324112064796),
    (52.09840031665556, 11.178141307841926),
    (52.16989765458063, 11.240398225811637),
    (52.09252130832008, 11.154333967570254),
    (52.13788959193653, 11.091422237662943),
    (52.141548691146795, 11.32608184518228),
    (52.09563438305892, 11.293991719352936),
    (52.10089941396949, 11.298974658146317),
    (52.0935070621907, 11.139002406753155),
    (52.15992543802094, 11.327600135322013),
    (52.15837860999751, 11.168890129175152),
    (52.118178728522054, 11.317112555354184),
    (52.09587893233216, 11.27745661214289),
    (52.07480910547546, 11.094384694737808),
    (52.14833123951739, 11.127341727137027),
    (52.1278198204315, 11.340525155358376),
    (52.165742943249334, 11.304459361752187),
    (52.15692596390275, 11.1628456676591),
    (52.12028794270609, 11.008870428520757),
    (52.097752824389396, 11.195809386685433),
    (52.15072202658883, 11.325284574975335),
],
```
Suche in gegebenem AOI nach Städten und urbanen Gebiete und stelle diese als neuen Layer “urbaneGebiete” auf dem SitaWare Plan “LayerTest” dar! Bleibe so minimal wie möglich: Query die urbanen Gebiete durch einen einzigen Call (als Task) in eine .geojson file und in einem zweiten task(write-oplan) schreibe diese .geojson file in den Plan! Mehr als zwei sequentielle Tasks dürfen nicht notwendig sein!!!

"""

TASK_NAME = "llm_geo_task"
# Register provider tools here. Every tool must materialize GeoJSON in the run data
# directory and follow llm_geo.tools.data_retrieval.provider_tool_instructions().
RETRIEVAL_TOOLS: list[BaseTool] = PUBLIC_RETRIEVAL_TOOLS
# Import modules containing @code functions before this line, then expose them here.
REGISTERED_OPERATIONS = registered_operations()

# Durable runtime configuration belongs in .env. Keep task-specific inputs below or
# pass them through the CLI. An empty model enables the offline readiness check.
MODEL = os.getenv("LLM_GEO_MODEL", "gpt-5.4-mini").strip()
MODEL_PROVIDER = os.getenv("LLM_GEO_MODEL_PROVIDER", "openai").strip() or None
BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

# Graph mode is more robust. Direct mode skips DAG and operation decomposition.
DIRECT_MODE = _environment_bool("LLM_GEO_DIRECT_MODE", False)

# Optional conversational supervisor around the complete LLM-GEO workflow.
USE_DEEP_AGENT = _environment_bool("LLM_GEO_USE_DEEP_AGENT", False)

# Generated Python runs only when enabled.
ALLOW_CODE_EXECUTION = _environment_bool("LLM_GEO_ALLOW_CODE_EXECUTION", True)

# Every run: OUTPUT_ROOT / TASK_NAME / UTC_TIMESTAMP.
OUTPUT_ROOT = Path(os.getenv("LLM_GEO_OUTPUT_ROOT", "output"))

# Bounded autonomous correction.
MAX_PLAN_ATTEMPTS = _environment_positive_int("LLM_GEO_MAX_PLAN_ATTEMPTS", 3)
MAX_EXECUTION_ATTEMPTS = _environment_positive_int("LLM_GEO_MAX_EXECUTION_ATTEMPTS", 10)

# INFO: concise progress. DEBUG: additional file detail.
LOG_LEVEL = _environment_log_level("LLM_GEO_LOG_LEVEL", logging.INFO)
LOG_HTTP = _environment_bool("LLM_GEO_LOG_HTTP", True)


def initialize_model():
    model_name = os.getenv("LLM_GEO_MODEL", "gpt-5.4-mini").strip()
    if not model_name:
        raise RuntimeError("LLM_GEO_MODEL is empty")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    model = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.3,
        timeout=720,
        **{"extra_body": {"cache": {"no-cache": True}, "configurable": {"stream": False}}}
    )

    return model


def main(task: str = TASK, task_name: str = TASK_NAME) -> None:
    """Execute the configured task or its command-line override."""
    configure_logging(LOG_LEVEL, log_http=LOG_HTTP)
    logger = get_logger()
    if not task or not MODEL:
        logger.info("LLM-GEO ready | provider_connection=disabled")
        logger.info("Set a task and LLM_GEO_MODEL to run")
        return

    logger.info(
        "Initializing model | identifier=%s | provider=%s | endpoint=%s",
        MODEL,
        MODEL_PROVIDER or "inferred",
        "custom" if BASE_URL else "provider-default",
    )
    model = initialize_model()
    if USE_DEEP_AGENT:
        logger.info("Execution path | deep_agent=enabled")
        agent = create_geo_agent(
            model,
            retrieval_tools=RETRIEVAL_TOOLS,
            registered_operations=REGISTERED_OPERATIONS,
            default_task_name=task_name,
            output_root=OUTPUT_ROOT,
            direct_mode=DIRECT_MODE,
            allow_code_execution=ALLOW_CODE_EXECUTION,
            max_plan_attempts=MAX_PLAN_ATTEMPTS,
            max_execution_attempts=MAX_EXECUTION_ATTEMPTS,
            log_level=LOG_LEVEL,
            log_http=LOG_HTTP,
        )
        result = run_geo_agent(agent, task, task_name)
    else:
        logger.info("Execution path | deep_agent=disabled")
        result = run_llm_geo(
            model=model,
            task=task,
            task_name=task_name,
            retrieval_tools=RETRIEVAL_TOOLS,
            registered_operations=REGISTERED_OPERATIONS,
            output_root=OUTPUT_ROOT,
            direct_mode=DIRECT_MODE,
            allow_code_execution=ALLOW_CODE_EXECUTION,
            max_plan_attempts=MAX_PLAN_ATTEMPTS,
            max_execution_attempts=MAX_EXECUTION_ATTEMPTS,
            log_level=LOG_LEVEL,
            log_http=LOG_HTTP,
        )
    logger.info(
        "Run finished | status=%s | output=%s",
        result.get("status"),
        result.get("save_dir"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-GEO task.")
    parser.add_argument("--task", default=TASK, help="Geospatial task to execute.")
    parser.add_argument(
        "--task-name", default=TASK_NAME, help="Name used for the output directory."
    )
    arguments = parser.parse_args()
    main(task=arguments.task, task_name=arguments.task_name)
