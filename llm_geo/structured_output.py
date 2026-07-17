"""Structured-output strategy selection, configurable via LLM_STRUCTURED_OUTPUT_MODE.

"provider" forces the model's native JSON-schema response format (langchain's `ProviderStrategy`,
no tool calls). Some OpenAI-compatible endpoints (e.g. nemotron-ultra) don't support that strict
Structured Outputs API, only the older JSON-object response format -- "json_object" targets those:
any tool loop runs first with no response_format, then a separate completion call requests a single
JSON object matching the schema, validated locally with Pydantic (one retry on invalid JSON).
Tool-calling/function-calling is never used as the structured-output mechanism in either mode.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal, TypeVar

from deepagents import create_deep_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)
Mode = Literal["provider", "json_object"]


def structured_output_mode() -> Mode:
    mode = os.environ.get("LLM_STRUCTURED_OUTPUT_MODE", "json_object").strip().lower()
    if mode not in ("provider", "json_object"):
        raise ValueError(f"LLM_STRUCTURED_OUTPUT_MODE must be 'provider' or 'json_object', got {mode!r}")
    return mode  # type: ignore[return-value]


def run_structured_agent(
    model: BaseChatModel,
    system_prompt: str,
    user_content: str,
    schema: type[SchemaT],
    tools: list[BaseTool] | None = None,
) -> tuple[SchemaT, list[BaseMessage]]:
    """Run a (possibly tool-using) agent turn and return its validated structured output plus the
    full message transcript, so callers that retry on e.g. a failed contract test can inspect it."""
    tools = tools or []
    if structured_output_mode() == "provider":
        agent = create_deep_agent(
            model=model, tools=tools, system_prompt=system_prompt, response_format=ProviderStrategy(schema)
        )
        result = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
        return result["structured_response"], list(result["messages"])

    if not tools:
        return complete_json_object(model, system_prompt, user_content, schema), []

    agent = create_deep_agent(model=model, tools=tools, system_prompt=system_prompt)
    tool_result = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
    transcript = list(tool_result["messages"])

    context = user_content
    call_log = _tool_call_transcript(transcript)
    if call_log:
        context += "\n\nTOOL CALLS:\n" + "\n\n".join(call_log)
    final_text = _final_ai_text(transcript)
    if final_text:
        context += f"\n\nYour last message before finalizing:\n{final_text}"
    return complete_json_object(model, system_prompt, context, schema), transcript


def complete_json_object(
    model: BaseChatModel, system_prompt: str, user_content: str, schema: type[SchemaT]
) -> SchemaT:
    """A single tool-free completion constrained to JSON-object mode, validated against `schema`
    with one retry on invalid/non-conforming JSON."""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    json_model = model.bind(response_format={"type": "json_object"})
    correction = ""
    last_error: Exception | None = None
    for _ in range(2):
        response = json_model.invoke([
            SystemMessage(
                content=f"{system_prompt}\nReturn only one JSON object matching this JSON Schema: {schema_json}"
            ),
            HumanMessage(content=user_content + correction),
        ])
        try:
            return schema.model_validate(_extract_json_object(_message_text(response)))
        except (ValidationError, ValueError) as error:
            last_error = error
            correction = f"\n\nYour previous JSON was invalid. Return a corrected JSON object. Validation error: {error}"
    raise RuntimeError(f"json_object structured completion did not return a valid {schema.__name__}: {last_error}")


def _tool_call_transcript(messages: list[BaseMessage]) -> list[str]:
    """Pair each tool call's arguments with its result, in order (plain ToolMessage content alone
    loses the arguments -- e.g. contract_test's PASS/FAIL text without the code that produced it)."""
    calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls_by_id[call["id"]] = (call["name"], call["args"])
    entries = []
    for message in messages:
        if isinstance(message, ToolMessage):
            name, args = calls_by_id.get(message.tool_call_id, (message.name or "tool", {}))
            args_text = json.dumps(args, ensure_ascii=False, default=str)
            entries.append(f"Called {name}({args_text}) -> {message.content}")
    return entries


def _final_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return ""


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in message.content
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first valid JSON object, tolerating prose and Markdown fences around it."""
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
