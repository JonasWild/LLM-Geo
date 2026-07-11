"""Tests for durable generated-program revision and execution history."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_geo.tools.code_execution import (
    publish_solution,
    save_code_revision,
    save_execution_attempt,
    save_execution_result,
)


class CodeHistoryTests(unittest.TestCase):
    def test_revisions_are_numbered_immutably_and_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            save_dir = Path(temporary)
            first = save_code_revision(save_dir, "print('raw')\n", "assembler_raw")
            second = save_code_revision(
                save_dir,
                "print('reviewed')\n",
                "assembler_reviewed",
                parent_revision=first["revision"],
            )

            self.assertEqual(first["revision"], 1)
            self.assertEqual(second["revision"], 2)
            self.assertEqual(second["parent_revision"], 1)
            self.assertEqual(
                (save_dir / first["path"]).read_text(encoding="utf-8"),
                "print('raw')\n",
            )
            records = [
                json.loads(line)
                for line in (save_dir / "code" / "revisions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([record["revision"] for record in records], [1, 2])

    def test_revision_numbering_continues_from_files_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            save_dir = Path(temporary)
            save_code_revision(save_dir, "first", "assembler_raw")

            resumed = save_code_revision(save_dir, "second", "debugger")

            self.assertEqual(resumed["revision"], 2)
            self.assertTrue((save_dir / resumed["path"]).exists())

    def test_execution_attempt_is_idempotent_but_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            save_dir = Path(temporary)
            path = save_execution_attempt(save_dir, 1, "print('one')\n")

            self.assertEqual(
                save_execution_attempt(save_dir, 1, "print('one')\n"), path
            )
            with self.assertRaisesRegex(RuntimeError, "different code"):
                save_execution_attempt(save_dir, 1, "print('two')\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "print('one')\n")

    def test_execution_result_records_failure_and_disabled_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            save_dir = Path(temporary)
            save_execution_attempt(save_dir, 3, "raise RuntimeError('boom')\n")
            execution = {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": "Code execution is disabled.",
                "new_files": [],
            }

            record = save_execution_result(
                save_dir, 3, 7, execution, executed=False
            )

            persisted = json.loads(
                (save_dir / record["result_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["revision"], 7)
            self.assertFalse(persisted["executed"])
            self.assertFalse(persisted["success"])
            self.assertEqual(persisted["stderr"], "Code execution is disabled.")

    def test_published_solution_tracks_latest_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            save_dir = Path(temporary)
            publish_solution(save_dir, "print('first')\n")
            path = publish_solution(save_dir, "print('final')\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "print('final')\n")


if __name__ == "__main__":
    unittest.main()
