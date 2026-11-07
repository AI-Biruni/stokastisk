"""Wall-clock + CUDA-event timing for communication/compute segments. Segments are recorded; nothing is inferred."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from .metrics import summarize


class SegmentTimer:
    def __init__(self) -> None:
        self.segments: dict[str, list[float]] = {}

    @contextmanager
    def wall(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.segments.setdefault(name, []).append(time.perf_counter() - t)

    @contextmanager
    def cuda(self, name: str, stream=None):
        """GPU-side duration via CUDA events (synchronises the stream when the block exits)."""
        import torch
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(stream); 
        try:
            yield
        finally:
            b.record(stream); b.synchronize()
            self.segments.setdefault(name, []).append(a.elapsed_time(b) / 1000.0)

    def totals(self) -> dict[str, float]:
        return {k: sum(v) for k, v in self.segments.items()}

    def report(self) -> dict[str, Any]:
        return {k: summarize(v) for k, v in self.segments.items()}
