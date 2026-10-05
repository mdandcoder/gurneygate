# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Event log: stores no frames, only timestamped events (CSV).
Publication metrics come from here: lead time, false opens, missed stretchers.

Time has two columns: `ts` wall clock (processing time) and `t` scene time. Lead time
is computed from scene time: frame_index / source_fps for files, the monotonic clock live.
Measuring with the wall clock while processing a file measures processing speed, not the scene.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

HEADER = ["ts", "t", "event", "track_id", "tta", "occupied", "note", "run"]


class EventLog:
    def __init__(self, path: str | None, clock=None):
        self.path = Path(path) if path else None
        self.clock = clock or time.monotonic
        self.f = None
        self.run = time.strftime("%Y%m%d-%H%M%S")   # scene time restarts at 0 every run; the scorer separates runs
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with open(self.path, encoding="utf-8") as fh:
                    first = fh.readline().strip().split(",")
                if first != HEADER:     # old format: do not mix, move it aside
                    self.path.rename(self.path.with_suffix(self.path.suffix + ".old"))
            new = not self.path.exists()
            self.f = open(self.path, "a", newline="", encoding="utf-8")
            self.w = csv.writer(self.f)
            if new:
                self.w.writerow(HEADER)
        self.open_t: dict[int, float] = {}
        self.crossed: set[int] = set()

    def log(self, event, tid=None, tta=None, occupied=None, note=""):
        if not self.f:
            return
        with self._lock:
            self.w.writerow([f"{time.time():.3f}", f"{self.clock():.3f}", event, tid,
                             f"{tta:.2f}" if tta not in (None, float("inf")) else "", occupied, note, self.run])
            self.f.flush()

    def door_opened(self, tid, tta, occupied, extend=False):
        self.open_t.setdefault(tid, self.clock())
        self.log("door_extend" if extend else "door_open", tid, tta, occupied)

    def crossed_line(self, tid):
        """Stretcher crossed the threshold; lead time = crossing - trigger (scene time)."""
        if tid in self.crossed:
            return
        self.crossed.add(tid)
        lead = self.clock() - self.open_t[tid] if tid in self.open_t else None
        self.log("crossed", tid, note=f"lead={lead:.2f}s" if lead is not None else "no_trigger(missed)")

    def track_lost(self, tid):
        if tid in self.open_t and tid not in self.crossed:
            self.log("false_open?", tid, note="opened but never crossed")
        self.open_t.pop(tid, None)
        self.crossed.discard(tid)

    def close(self):
        if self.f:
            with self._lock:
                self.f.close()
                self.f = None
