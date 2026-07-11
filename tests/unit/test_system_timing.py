"""Tests for thresholded internal operation timing."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from llm_geo.system import timed_step


class TimedStepTests(unittest.TestCase):
    @patch("llm_geo.system.get_logger")
    @patch("llm_geo.system.time.perf_counter", side_effect=[10.0, 19.999])
    def test_fast_step_is_silent(self, _clock: Mock, get_logger: Mock) -> None:
        with timed_step("planner.call", 10.0):
            pass

        get_logger.assert_not_called()

    @patch("llm_geo.system.get_logger")
    @patch("llm_geo.system.time.perf_counter", side_effect=[10.0, 20.125])
    def test_slow_step_logs_duration_and_fields(
        self, _clock: Mock, get_logger: Mock
    ) -> None:
        logger = get_logger.return_value

        with timed_step("coder.call", 10.0, operation="buffer"):
            pass

        logger.warning.assert_called_once_with(
            "Slow step | step=%s%s | duration=%.3fs",
            "coder.call",
            " | operation=buffer",
            10.125,
        )

    @patch("llm_geo.system.get_logger")
    @patch("llm_geo.system.time.perf_counter", side_effect=[1.0, 12.0])
    def test_failed_slow_step_is_logged_and_reraised(
        self, _clock: Mock, get_logger: Mock
    ) -> None:
        logger = get_logger.return_value

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with timed_step("validator.call", 10.0):
                raise RuntimeError("boom")

        logger.warning.assert_called_once_with(
            "Slow step failed | step=%s%s | type=%s | duration=%.3fs",
            "validator.call",
            "",
            "RuntimeError",
            11.0,
        )


if __name__ == "__main__":
    unittest.main()
