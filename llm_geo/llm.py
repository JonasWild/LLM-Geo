"""Shared ChatOpenAI factory (the langchain-openai integration point)."""
from __future__ import annotations

import os

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

# Our agents never need the filesystem/shell/todo/subagent tooling deep agents ship by default --
# only structured output (planner) or a single contract_test tool (coder). Dropping the unused
# built-ins keeps every turn's prompt small, which matters a lot on low-tier OpenAI TPM limits.
register_harness_profile(
    "openai",
    HarnessProfile(
        excluded_tools=frozenset(
            {"write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
        ),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)


def get_model(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


retry_on_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=2, max=25),
    stop=stop_after_attempt(4),
    reraise=True,
)
