"""Concise, telegraphic tracing for every planning/impl/validation/repair/exec step."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("llm_geo")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


class Tracer:
    """Emits one telegraphic line per event and appends a JSONL record."""

    def __init__(self, path: str | Path = "traces/run.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, phase: str, node_id: str | None, status: str, dur_ms: float, **detail):
        rec = {"ts": time.time(), "phase": phase, "node_id": node_id, "status": status, "dur_ms": round(dur_ms, 1), **detail}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        extra = " ".join(f"{k}={v}" for k, v in detail.items())
        log.info("%-9s %-24s %-4s dur=%dms %s", phase.upper(), node_id or "-", status, dur_ms, extra)

    @contextmanager
    def span(self, phase: str, node_id: str | None = None, **detail):
        t0 = time.monotonic()
        try:
            yield
        except Exception as exc:
            self._write(phase, node_id, "ERR", (time.monotonic() - t0) * 1000, err=str(exc)[:120], **detail)
            raise
        else:
            self._write(phase, node_id, "OK", (time.monotonic() - t0) * 1000, **detail)
