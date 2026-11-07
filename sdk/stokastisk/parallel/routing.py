"""Weighted least-loaded routing (deterministic; ties go to the lowest worker id)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerLoad:
    id: int
    device: int
    alive: bool = True
    ready: bool = True
    pending_requests: int = 0
    pending_tokens: int = 0
    speed: float = 1.0                  # relative throughput weight, learned from observed latency (higher = faster)
    free_vram_mib: float | None = None  # last observation; used only to exclude a worker that is nearly out of memory
    min_free_vram_mib: float = 256.0
    speed_observed: bool = False

    @property
    def available(self) -> bool:
        if not (self.alive and self.ready):
            return False
        return self.free_vram_mib is None or self.free_vram_mib >= self.min_free_vram_mib


def projected_load(w: WorkerLoad, extra_tokens: int) -> float:
    """Estimated time-to-drain (arbitrary units): queued tokens + the new work, divided by the worker's speed."""
    return (w.pending_tokens + extra_tokens) / max(w.speed, 1e-6)


def choose_worker(workers: list[WorkerLoad], cost_tokens: int) -> int | None:
    cand = [w for w in workers if w.available]
    if not cand:
        return None
    return min(cand, key=lambda w: (projected_load(w, cost_tokens), w.pending_requests, w.id)).id
