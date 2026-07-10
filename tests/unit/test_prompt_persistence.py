"""Tests for meaningful, run-scoped LLM prompt filenames."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_geo.subagents.runtime import build_review_prompt
from llm_geo.utils.prompts import save_prompt


class PromptPersistenceTests(unittest.TestCase):
    def test_filename_identifies_sequence_stage_agent_subject_and_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = save_prompt(
                temporary_directory,
                stage="ops",
                agent="coder",
                subject="op create-map",
                prompt="first prompt",
            )
            second = save_prompt(
                temporary_directory,
                stage="ops",
                agent="coder",
                subject="op create-map",
                prompt="second prompt",
            )

            self.assertEqual(first.name, "001_code_op_create_map_01.txt")
            self.assertEqual(second.name, "002_code_op_create_map_02.txt")
            self.assertIn("Stage: ops", first.read_text(encoding="utf-8"))
            self.assertTrue(first.read_text(encoding="utf-8").endswith("first prompt"))

    def test_sequence_continues_across_different_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_prompt(
                temporary_directory,
                stage="retrieve",
                agent="retriever",
                subject="sources",
                prompt="retrieve",
            )
            planner = save_prompt(
                temporary_directory,
                stage="plan",
                agent="planner",
                subject="workflow",
                prompt="plan",
            )

            self.assertEqual(planner.name, "002_plan_01.txt")

    def test_compact_names_still_distinguish_specialized_reviews(self) -> None:
        cases = [
            ("assemble", "assembler", "program", "001_assemble_01.txt"),
            ("assemble", "reviewer", "program", "001_review_assemble_01.txt"),
            ("direct", "coder", "program", "001_direct_01.txt"),
            ("direct", "reviewer", "program", "001_review_direct_01.txt"),
            ("debug", "debugger", "program", "001_debug_01.txt"),
            ("validate", "validator", "result", "001_validate_01.txt"),
        ]
        for stage, agent, subject, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = save_prompt(
                        temporary_directory,
                        stage=stage,
                        agent=agent,
                        subject=subject,
                        prompt="prompt",
                    )
                    self.assertEqual(path.name, expected)

    def test_saved_reviewer_prompt_can_match_submitted_text(self) -> None:
        prompt = build_review_prompt("def solve(): pass", "Implement solve.")

        self.assertEqual(
            prompt,
            "Review this Python code against the requirements.\n\n"
            "REQUIREMENTS:\nImplement solve.\n\nCODE:\ndef solve(): pass",
        )

    def test_long_subjects_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = save_prompt(
                temporary_directory,
                stage="ops",
                agent="coder",
                subject="operation_" + "a" * 60,
                prompt="first",
            )
            second = save_prompt(
                temporary_directory,
                stage="ops",
                agent="coder",
                subject="operation_" + "b" * 60,
                prompt="second",
            )

            self.assertNotEqual(
                first.name.rsplit("_01.txt", 1)[0],
                second.name.rsplit("_01.txt", 1)[0],
            )


if __name__ == "__main__":
    unittest.main()
