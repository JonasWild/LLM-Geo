"""LLM-GEO task configuration and executable entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool

from llm_geo.middleware.logging import configure_logging, get_logger
from llm_geo.operations import registered_operations
import llm_geo.operations.basic
from llm_geo.subagents.supervisor import create_geo_agent, run_geo_agent
from llm_geo.system import run_llm_geo
from llm_geo.tools.public_data_providers import PUBLIC_RETRIEVAL_TOOLS

import dotenv
dotenv.load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK = "Finde die urbane Gebiete mittels overpass zwischen Magedburg und Hohes Holz und stelle sie als png dar."
TASK_NAME = "llm_geo_task"
# Register provider tools here. Every tool must materialize GeoJSON in the run data
# directory and follow llm_geo.tools.data_retrieval.provider_tool_instructions().
RETRIEVAL_TOOLS: list[BaseTool] = PUBLIC_RETRIEVAL_TOOLS
# Import modules containing @code functions before this line, then expose them here.
REGISTERED_OPERATIONS = registered_operations()

# Model example: "openai:gpt-4o". Empty = offline readiness check.
MODEL = "gpt-5.4-mini"

# Graph mode is more robust. Direct mode skips DAG and operation decomposition.
DIRECT_MODE = False

# Optional conversational supervisor around the complete LLM-GEO workflow.
USE_DEEP_AGENT = False

# Generated Python runs only when enabled.
ALLOW_CODE_EXECUTION = True

# Every run: OUTPUT_ROOT / TASK_NAME / UTC_TIMESTAMP.
OUTPUT_ROOT = Path("output")

# Bounded autonomous correction.
MAX_PLAN_ATTEMPTS = 3
MAX_EXECUTION_ATTEMPTS = 10

# INFO: concise progress. DEBUG: additional file detail.
LOG_LEVEL = logging.INFO


def main(task: str = TASK, task_name: str = TASK_NAME) -> None:
    """Execute the configured task or its command-line override."""
    configure_logging(LOG_LEVEL)
    logger = get_logger()
    if not task or not MODEL:
        logger.info("LLM-GEO ready | provider_connection=disabled")
        logger.info("Set TASK, MODEL, and RETRIEVAL_TOOLS in main.py to run")
        return

    logger.info("Initializing model | identifier=%s", MODEL)
    model = init_chat_model(MODEL)
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
