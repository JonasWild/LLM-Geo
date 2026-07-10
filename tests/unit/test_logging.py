"""Tests for application and opt-in HTTP request logging."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_geo.middleware.logging import close_file_logging, configure_logging


class HttpLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        with patch.dict(os.environ, {"LLM_GEO_LOG_HTTP": "false"}):
            configure_logging(logging.INFO)

    def test_http_client_records_are_written_to_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "llm_geo.log"
            try:
                with patch.dict(os.environ, {"LLM_GEO_LOG_HTTP": "true"}):
                    configure_logging(logging.INFO, log_path)

                logging.getLogger("httpx").info(
                    'HTTP Request: POST https://llm.example/v1/chat/completions "200 OK"'
                )
                logging.getLogger("urllib3.connectionpool").debug(
                    'https://osm.example:443 "GET /search HTTP/1.1" 200'
                )

                contents = log_path.read_text(encoding="utf-8")
                self.assertIn("https://llm.example/v1/chat/completions", contents)
                self.assertIn("httpx", contents)
                self.assertIn("https://osm.example", contents)
                self.assertIn("urllib3.connectionpool", contents)
            finally:
                close_file_logging()

    def test_http_logging_is_enabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configure_logging(logging.DEBUG)

        self.assertEqual(logging.getLogger("httpx").level, logging.INFO)
        self.assertEqual(
            logging.getLogger("urllib3.connectionpool").level, logging.DEBUG
        )

    def test_http_logging_can_be_disabled_explicitly(self) -> None:
        configure_logging(logging.DEBUG, log_http=False)

        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(
            logging.getLogger("urllib3.connectionpool").level, logging.WARNING
        )

    def test_invalid_http_logging_value_is_rejected(self) -> None:
        with patch.dict(os.environ, {"LLM_GEO_LOG_HTTP": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "LLM_GEO_LOG_HTTP"):
                configure_logging(logging.INFO)


if __name__ == "__main__":
    unittest.main()
