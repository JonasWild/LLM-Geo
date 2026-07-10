"""Small LangChain agent helpers shared across workflow stages."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from llm_geo.middleware.logging import get_logger
from llm_geo.utils.models import ReviewDecision


def create_structured_agent(
    model: BaseChatModel,
    system_prompt: str,
    schema: type[BaseModel],
    tools: list[BaseTool] | None = None,
) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=tools or [],
        system_prompt=system_prompt,
        response_format=schema,
    )


def ask_structured(agent: CompiledStateGraph, prompt: str) -> BaseModel:
    response = _invoke_structured(agent, prompt)
    result = response.get("structured_response")
    if result is None:
        raise RuntimeError("Agent returned no structured_response")
    return result


def ask_structured_with_tool_results(
    agent: CompiledStateGraph, prompt: str
) -> tuple[BaseModel, list[str]]:
    """Return a structured response plus raw results from tools used by the agent."""
    response = _invoke_structured(agent, prompt)
    result = response.get("structured_response")
    if result is None:
        raise RuntimeError("Agent returned no structured_response")
    tool_results = [
        str(message.content)
        for message in response.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    return result, tool_results


def _invoke_structured(agent: CompiledStateGraph, prompt: str) -> dict[str, object]:
    request = {"messages": [{"role": "user", "content": prompt}]}
    for attempt in range(2):
        try:
            return agent.invoke(request)
        except StructuredOutputValidationError:
            if attempt:
                raise
            get_logger().warning("Structured output was invalid | retrying once")
    raise RuntimeError("Structured agent did not return a response")


def review_code(
    reviewer: CompiledStateGraph,
    code: str,
    requirements: str,
) -> tuple[str, list[str]]:
    decision = ask_structured(
        reviewer,
        f"Review this Python code against the requirements.\n\n"
        f"REQUIREMENTS:\n{requirements}\n\nCODE:\n{code}",
    )
    if not isinstance(decision, ReviewDecision):
        raise TypeError("Reviewer returned an unexpected response type")
    if decision.passed:
        get_logger().debug("Code review passed without changes")
        return code, decision.issues
    if not decision.corrected_code:
        raise RuntimeError(
            f"Code review failed without corrected code: {decision.issues}"
        )
    get_logger().info("Code review corrected %d issue(s)", len(decision.issues))
    return decision.corrected_code, decision.issues
