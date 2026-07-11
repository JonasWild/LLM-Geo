"""Small LangChain agent helpers shared across workflow stages."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from llm_geo.middleware.logging import get_logger
from llm_geo.utils.models import ReviewDecision


class StructuredAgent(Protocol):
    """Minimal interface shared by LangGraph and JSON-mode structured agents."""

    def invoke(self, request: dict[str, object]) -> dict[str, object]: ...


class PromptedStructuredAgent:
    """Prompt for JSON text and validate it locally with Pydantic."""

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        schema: type[BaseModel],
        tools: list[BaseTool],
        *,
        json_mode: bool = False,
    ) -> None:
        self._model = (
            model.bind(response_format={"type": "json_object"})
            if json_mode
            else model
        )
        self._system_prompt = system_prompt
        self._schema = schema
        self._tool_agent = (
            create_agent(model=model, tools=tools, system_prompt=system_prompt)
            if tools
            else None
        )

    def invoke(self, request: dict[str, object]) -> dict[str, object]:
        messages = request.get("messages", [])
        prompt = _last_user_content(messages)
        agent_messages: list[Any] = []
        tool_results: list[str] = []
        if self._tool_agent is not None:
            tool_response = self._tool_agent.invoke(request)
            agent_messages = list(tool_response.get("messages", []))
            tool_results = [
                str(message.content)
                for message in agent_messages
                if isinstance(message, ToolMessage)
            ]

        context = prompt
        if tool_results:
            context += "\n\nTOOL RESULTS:\n" + "\n\n".join(tool_results)
        elif self._tool_agent is not None:
            context += "\n\nNo provider tool returned a result."

        schema_json = json.dumps(self._schema.model_json_schema(), ensure_ascii=False)
        correction = ""
        for attempt in range(2):
            response = self._model.invoke(
                [
                    SystemMessage(
                        content=(
                            self._system_prompt
                            + "\nReturn only one JSON object matching this JSON Schema: "
                            + schema_json
                        )
                    ),
                    HumanMessage(content=context + correction),
                ]
            )
            content = _message_text(response)
            try:
                result = self._schema.model_validate(_extract_json_object(content))
                return {
                    "structured_response": result,
                    "messages": [*agent_messages, response],
                }
            except (ValidationError, ValueError) as error:
                if attempt:
                    raise StructuredOutputValidationError(
                        self._schema.__name__, error, response
                    ) from error
                get_logger().warning("Structured output was invalid | retrying once")
                correction = (
                    "\n\nYour previous JSON was invalid. Return a corrected JSON object. "
                    f"Validation error: {error}"
                )
        raise RuntimeError("JSON-mode structured agent did not return a response")


class JsonModeStructuredAgent(PromptedStructuredAgent):
    """Use provider JSON mode, then validate the response locally."""

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        schema: type[BaseModel],
        tools: list[BaseTool],
    ) -> None:
        super().__init__(model, system_prompt, schema, tools, json_mode=True)


def _last_user_content(messages: object) -> str:
    if not isinstance(messages, list):
        raise TypeError("Structured agent messages must be a list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("Structured agent request contains no user message")


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        str(block.get("text", "")) if isinstance(block, dict) else str(block)
        for block in message.content
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first valid JSON object, tolerating prose and Markdown fences."""
    decoder = json.JSONDecoder()
    for position, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response contained no valid JSON object")


def _structured_output_strategy() -> str:
    strategy = os.getenv("LLM_GEO_STRUCTURED_OUTPUT", "auto").strip().lower()
    if strategy not in {"auto", "tool", "provider", "json_mode", "prompted"}:
        raise ValueError(
            "LLM_GEO_STRUCTURED_OUTPUT must be auto, tool, provider, json_mode, "
            "or prompted"
        )
    return strategy


def create_structured_agent(
    model: BaseChatModel,
    system_prompt: str,
    schema: type[BaseModel],
    tools: list[BaseTool] | None = None,
) -> StructuredAgent:
    strategy = _structured_output_strategy()
    selected_tools = tools or []
    if strategy == "json_mode":
        return JsonModeStructuredAgent(model, system_prompt, schema, selected_tools)
    if strategy == "prompted":
        return PromptedStructuredAgent(model, system_prompt, schema, selected_tools)
    return create_agent(
        model=model,
        tools=selected_tools,
        system_prompt=system_prompt,
        response_format=(
            ProviderStrategy(schema)
            if strategy == "provider"
            else ToolStrategy(schema)
            if strategy == "tool"
            else schema
        ),
    )


def ask_structured(agent: StructuredAgent, prompt: str) -> BaseModel:
    response = _invoke_structured(agent, prompt)
    result = response.get("structured_response")
    if result is None:
        raise RuntimeError("Agent returned no structured_response")
    return result


def ask_structured_with_tool_results(
    agent: StructuredAgent, prompt: str
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


def _invoke_structured(agent: StructuredAgent, prompt: str) -> dict[str, object]:
    request = {"messages": [{"role": "user", "content": prompt}]}
    if isinstance(agent, PromptedStructuredAgent):
        return agent.invoke(request)
    for attempt in range(2):
        try:
            return agent.invoke(request)
        except StructuredOutputValidationError:
            if attempt:
                raise
            get_logger().warning("Structured output was invalid | retrying once")
    raise RuntimeError("Structured agent did not return a response")


def review_code(
    reviewer: StructuredAgent,
    code: str,
    requirements: str,
    *,
    prompt: str | None = None,
) -> tuple[str, list[str]]:
    decision = ask_structured(reviewer, prompt or build_review_prompt(code, requirements))
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


def build_review_prompt(code: str, requirements: str) -> str:
    """Build the exact user prompt submitted to the code reviewer."""
    return (
        "Review this Python code against the requirements.\n\n"
        f"REQUIREMENTS:\n{requirements}\n\nCODE:\n{code}"
    )
