"""Concise, telegraphic tracing for every planning/impl/validation/repair/exec step.

The console line stays short (errors truncated); the JSONL record keeps the full traceback on
errors, and an optional `on_error` hook lets a RunArtifacts bundle collect every error centrally.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

log = logging.getLogger("llm_geo")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

# on_error(phase, node_id, message, traceback_text)
OnError = Callable[[str, str | None, str, str], None]


class Tracer:
    """Emits one telegraphic line per event and appends a JSONL record."""

    def __init__(
        self,
        path: str | Path = "traces/run.jsonl",
        log_file: str | Path | None = None,
        on_error: OnError | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.on_error = on_error

    def _write(self, phase: str, node_id: str | None, status: str, dur_ms: float,
               traceback_text: str | None = None, **detail):
        rec = {"ts": time.time(), "phase": phase, "node_id": node_id, "status": status, "dur_ms": round(dur_ms, 1), **detail}
        if traceback_text:
            rec["traceback"] = traceback_text
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        extra = " ".join(f"{k}={v}" for k, v in detail.items())
        line = f"{phase.upper():<9} {node_id or '-':<24} {status:<4} dur={int(dur_ms)}ms {extra}"
        log.info("%s", line)
        if self.log_file:
            with self.log_file.open("a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    @contextmanager
    def span(self, phase: str, node_id: str | None = None, **detail):
        t0 = time.monotonic()
        try:
            yield
        except Exception as exc:
            tb = traceback.format_exc()
            self._write(phase, node_id, "ERR", (time.monotonic() - t0) * 1000,
                        traceback_text=tb, err=str(exc)[:120], **detail)
            if self.on_error:
                self.on_error(phase, node_id, f"{type(exc).__name__}: {exc}", tb)
            raise
        else:
            self._write(phase, node_id, "OK", (time.monotonic() - t0) * 1000, **detail)
