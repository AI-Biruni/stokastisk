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


def effective_speed(w: WorkerLoad, reference_speed: float | None = None) -> float:
    """Speed used for routing. An observed speed is tokens/s; the default 1.0 is unit-less, so an unobserved worker is
    given the reference speed (mean of the observed workers) instead. Otherwise one worker that has learned its speed
    would look orders of magnitude faster than its unobserved peer and monopolise all work."""
    if w.speed_observed or reference_speed is None:
        return max(w.speed, 1e-6)
    return max(reference_speed, 1e-6)


def projected_load(w: WorkerLoad, extra_tokens: int, reference_speed: float | None = None) -> float:
    """Estimated time-to-drain (arbitrary units): queued tokens + the new work, divided by the worker's speed."""
    return (w.pending_tokens + extra_tokens) / effective_speed(w, reference_speed)


def choose_worker(workers: list[WorkerLoad], cost_tokens: int) -> int | None:
    cand = [w for w in workers if w.available]
    if not cand:
        return None
    observed = [w.speed for w in workers if w.speed_observed and w.alive]
    ref = sum(observed) / len(observed) if observed else None
    return min(cand, key=lambda w: (projected_load(w, cost_tokens, ref), w.pending_requests, w.id)).id
