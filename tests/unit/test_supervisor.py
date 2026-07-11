"""Offline tests for the optional Deep Agents supervisor path."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import main as main_module
from llm_geo.subagents.supervisor import create_geo_agent, run_geo_agent


class _ToolCallingModel(BaseChatModel):
    """Return one workflow tool call followed by a final supervisor response."""

    _response_index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "offline-tool-calling-model"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._response_index == 0:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_geospatial_analysis",
                        "args": {
                            "task": "Map the parks",
                            "task_name": "model_selected_name",
                        },
                        "id": "workflow-call",
                    }
                ],
            )
        else:
            message = AIMessage(content="The verified analysis completed.")
        self._response_index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class SupervisorTests(unittest.TestCase):
    @patch("llm_geo.subagents.supervisor.run_llm_geo")
    def test_supervisor_forwards_configuration_and_returns_summary(
        self, run_workflow: Mock
    ) -> None:
        run_workflow.return_value = {
            "status": "complete",
            "save_dir": "output/configured/run",
            "execution": {"success": True},
            "validation": {"valid": True},
            "artifacts": ["results/map.png"],
        }
        model = _ToolCallingModel()
        retrieval_tools: list[Any] = []
        registered_operations: list[Any] = []
        agent = create_geo_agent(
            model,
            retrieval_tools=retrieval_tools,
            registered_operations=registered_operations,
            default_task_name="configured_name",
            output_root=Path("custom-output"),
            direct_mode=True,
            allow_code_execution=False,
            max_plan_attempts=4,
            max_execution_attempts=5,
            log_level=logging.DEBUG,
            generate_mermaid=False,
            slow_step_seconds=2.5,
        )

        result = run_geo_agent(agent, "Map the parks", "configured_name")

        run_workflow.assert_called_once_with(
            model,
            "Map the parks",
            "configured_name",
            retrieval_tools=retrieval_tools,
            registered_operations=registered_operations,
            output_root=Path("custom-output"),
            direct_mode=True,
            allow_code_execution=False,
            max_plan_attempts=4,
            max_execution_attempts=5,
            log_level=logging.DEBUG,
            log_http=True,
            generate_mermaid=False,
            slow_step_seconds=2.5,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["save_dir"], "output/configured/run")
        self.assertEqual(result["final_message"], "The verified analysis completed.")

    @patch.object(main_module, "configure_logging")
    @patch.object(main_module, "get_logger")
    @patch.object(main_module, "initialize_model")
    @patch.object(main_module, "run_geo_agent")
    @patch.object(main_module, "create_geo_agent")
    @patch.object(main_module, "run_llm_geo")
    def test_main_uses_deep_agent_when_enabled(
        self,
        run_workflow: Mock,
        create_agent: Mock,
        run_agent: Mock,
        initialize_model: Mock,
        get_logger: Mock,
        configure_logging: Mock,
    ) -> None:
        model = object()
        compiled_agent = object()
        initialize_model.return_value = model
        create_agent.return_value = compiled_agent
        run_agent.return_value = {
            "status": "complete",
            "save_dir": "output/deep/run",
        }
        get_logger.return_value = Mock()

        with patch.object(main_module, "USE_DEEP_AGENT", True):
            main_module.main(task="Map parks", task_name="parks")

        run_workflow.assert_not_called()
        create_agent.assert_called_once_with(
            model,
            retrieval_tools=main_module.RETRIEVAL_TOOLS,
            registered_operations=main_module.REGISTERED_OPERATIONS,
            default_task_name="parks",
            output_root=main_module.OUTPUT_ROOT,
            direct_mode=main_module.DIRECT_MODE,
            allow_code_execution=main_module.ALLOW_CODE_EXECUTION,
            max_plan_attempts=main_module.MAX_PLAN_ATTEMPTS,
            max_execution_attempts=main_module.MAX_EXECUTION_ATTEMPTS,
            log_level=main_module.LOG_LEVEL,
            log_http=main_module.LOG_HTTP,
            generate_mermaid=main_module.GENERATE_MERMAID,
            slow_step_seconds=main_module.SLOW_STEP_SECONDS,
        )
        run_agent.assert_called_once_with(compiled_agent, "Map parks", "parks")

    @patch.object(main_module, "configure_logging")
    @patch.object(main_module, "get_logger")
    @patch.object(main_module, "init_chat_model")
    @patch.object(main_module, "run_geo_agent")
    @patch.object(main_module, "create_geo_agent")
    @patch.object(main_module, "run_llm_geo")
    def test_main_uses_workflow_directly_when_supervisor_is_disabled(
        self,
        run_workflow: Mock,
        create_agent: Mock,
        run_agent: Mock,
        init_model: Mock,
        get_logger: Mock,
        configure_logging: Mock,
    ) -> None:
        init_model.return_value = object()
        run_workflow.return_value = {
            "status": "complete",
            "save_dir": "output/direct/run",
        }
        get_logger.return_value = Mock()

        with patch.object(main_module, "USE_DEEP_AGENT", False):
            main_module.main(task="Map parks", task_name="parks")

        run_workflow.assert_called_once()
        create_agent.assert_not_called()
        run_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
