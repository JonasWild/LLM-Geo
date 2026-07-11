"""Tests for selectable structured-output strategies."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from llm_geo.subagents.runtime import (
    JsonModeStructuredAgent,
    ask_structured,
    create_structured_agent,
)


class ExampleResult(BaseModel):
    answer: str


class StructuredOutputTests(unittest.TestCase):
    def test_json_mode_binds_response_format_and_validates_json(self) -> None:
        parser = MagicMock()
        parser.invoke.return_value = AIMessage(content='{"answer": "ok"}')
        model = MagicMock()
        model.bind.return_value = parser

        with patch.dict(os.environ, {"LLM_GEO_STRUCTURED_OUTPUT": "json_mode"}):
            agent = create_structured_agent(model, "System", ExampleResult)
            result = ask_structured(agent, "Question")

        self.assertIsInstance(agent, JsonModeStructuredAgent)
        self.assertEqual(result, ExampleResult(answer="ok"))
        model.bind.assert_called_once_with(
            response_format={"type": "json_object"}
        )

    @patch("llm_geo.subagents.runtime.create_agent")
    def test_provider_strategy_is_explicit(self, create_agent: MagicMock) -> None:
        model = MagicMock()
        with patch.dict(os.environ, {"LLM_GEO_STRUCTURED_OUTPUT": "provider"}):
            create_structured_agent(model, "System", ExampleResult)

        response_format = create_agent.call_args.kwargs["response_format"]
        self.assertIsInstance(response_format, ProviderStrategy)
        self.assertIs(response_format.schema, ExampleResult)

    def test_unknown_strategy_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {"LLM_GEO_STRUCTURED_OUTPUT": "unknown"}),
            self.assertRaisesRegex(ValueError, "LLM_GEO_STRUCTURED_OUTPUT"),
        ):
            create_structured_agent(MagicMock(), "System", ExampleResult)


if __name__ == "__main__":
    unittest.main()
