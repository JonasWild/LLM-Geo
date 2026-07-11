"""Tests for environment-backed entry-point configuration."""

from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import Mock, patch

import main as main_module


class EnvironmentConfigurationTests(unittest.TestCase):
    def test_boolean_parser_accepts_common_values(self) -> None:
        for raw_value in ("1", "true", "YES", "on"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"SETTING": raw_value}
            ):
                self.assertTrue(main_module._environment_bool("SETTING", False))

        for raw_value in ("0", "false", "NO", "off"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"SETTING": raw_value}
            ):
                self.assertFalse(main_module._environment_bool("SETTING", True))

    def test_invalid_environment_values_are_rejected(self) -> None:
        with patch.dict(os.environ, {"SETTING": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "SETTING"):
                main_module._environment_bool("SETTING", False)

        with patch.dict(os.environ, {"SETTING": "0"}):
            with self.assertRaisesRegex(ValueError, "SETTING"):
                main_module._environment_positive_int("SETTING", 3)

        with patch.dict(os.environ, {"SETTING": "verbose-ish"}):
            with self.assertRaisesRegex(ValueError, "SETTING"):
                main_module._environment_log_level("SETTING", logging.INFO)

        for raw_value in ("0", "-1", "slow", "nan", "inf"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"SETTING": raw_value}
            ):
                with self.assertRaisesRegex(ValueError, "SETTING"):
                    main_module._environment_positive_float("SETTING", 10.0)

    def test_positive_float_parser_accepts_fractional_seconds(self) -> None:
        with patch.dict(os.environ, {"SETTING": "2.5"}):
            self.assertEqual(
                main_module._environment_positive_float("SETTING", 10.0), 2.5
            )

    @patch.object(main_module, "ChatOpenAI")
    def test_model_initializer_passes_model_and_custom_base_url(
        self, chat_model: Mock
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_GEO_MODEL": "local-model",
                "OPENAI_BASE_URL": "http://localhost:1234/v1",
                "OPENAI_API_KEY": "secret",
            },
        ):
            main_module.initialize_model()

        chat_model.assert_called_once_with(
            base_url="http://localhost:1234/v1",
            api_key="secret",
            model="local-model",
            temperature=0.3,
            timeout=720,
            extra_body={
                "cache": {"no-cache": True},
                "configurable": {"stream": False},
            },
        )

    def test_model_initializer_rejects_empty_model(self) -> None:
        with patch.dict(os.environ, {"LLM_GEO_MODEL": ""}):
            with self.assertRaisesRegex(RuntimeError, "LLM_GEO_MODEL is empty"):
                main_module.initialize_model()


if __name__ == "__main__":
    unittest.main()
