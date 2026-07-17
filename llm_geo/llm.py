"""Shared ChatOpenAI/OpenAIEmbeddings factories (the langchain-openai integration point).

Model, base URL, and API key are all configurable via env vars -- see .env.example.
"""
from __future__ import annotations

import os

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
    kwargs: dict = {"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": temperature}
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    if api_key := os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    kwargs: dict = {"model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")}
    # Embedding-specific overrides fall back to the shared LLM base_url/api_key so a single-provider
    # setup only needs to set OPENAI_BASE_URL/OPENAI_API_KEY once.
    if base_url := os.environ.get("OPENAI_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    if api_key := os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


retry_on_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=2, max=25),
    stop=stop_after_attempt(4),
    reraise=True,
)
