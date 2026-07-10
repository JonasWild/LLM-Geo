"""Tests for environment-backed entry-point configuration."""

from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

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

    def test_model_initializer_passes_provider_and_custom_base_url(self) -> None:
        with (
            patch.object(main_module, "MODEL", "local-model"),
            patch.object(main_module, "MODEL_PROVIDER", "openai"),
            patch.object(main_module, "BASE_URL", "http://localhost:1234/v1"),
            patch.object(main_module, "init_chat_model") as init_model,
        ):
            main_module._initialize_model()

        init_model.assert_called_once_with(
            "local-model",
            model_provider="openai",
            base_url="http://localhost:1234/v1",
            use_responses_api=False,
        )

    def test_model_initializer_omits_empty_optional_settings(self) -> None:
        with (
            patch.object(main_module, "MODEL", "openai:gpt-4o"),
            patch.object(main_module, "MODEL_PROVIDER", None),
            patch.object(main_module, "BASE_URL", None),
            patch.object(main_module, "init_chat_model") as init_model,
        ):
            main_module._initialize_model()

        init_model.assert_called_once_with("openai:gpt-4o")


if __name__ == "__main__":
    unittest.main()
