"""Deterministic, token-aware, weighted least-loaded scheduler. Pure Python: no CUDA, no processes (the pool owns those)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .queue import Request
from .routing import WorkerLoad, choose_worker


@dataclass
class Decision:
    request_id: int
    worker_id: int | None
    cost: int
    loads_before: dict[int, float] = field(default_factory=dict)


class Scheduler:
    def __init__(self, workers: list[WorkerLoad], *, max_attempts: int = 2, speed_alpha: float = 0.3) -> None:
        self.workers = {w.id: w for w in workers}
        self.max_attempts = max_attempts; self.alpha = speed_alpha
        self.inflight: dict[int, tuple[int, Request]] = {}     # request id -> (worker id, request)
        self.log: list[Decision] = []

    def assign(self, req: Request) -> int | None:
        before = {w.id: w.pending_tokens for w in self.workers.values()}
        wid = choose_worker(list(self.workers.values()), req.cost)
        self.log.append(Decision(req.id, wid, req.cost, before))
        if wid is None:
            return None
        w = self.workers[wid]; w.pending_requests += 1; w.pending_tokens += req.cost
        req.attempts += 1; self.inflight[req.id] = (wid, req)
        return wid

    def assign_batch(self, reqs: list[Request], *, longest_first: bool = True) -> dict[int, list[Request]]:
        """Greedy LPT (longest-processing-time-first) placement; deterministic. Returns worker id -> requests; unplaceable ones under key -1."""
        order = sorted(reqs, key=lambda r: (-r.cost, r.id)) if longest_first else list(reqs)
        out: dict[int, list[Request]] = {}
        for r in order:
            wid = self.assign(r)
            out.setdefault(wid if wid is not None else -1, []).append(r)
        return out

    def complete(self, req_id: int, *, seconds: float | None = None) -> None:
        wid, req = self.inflight.pop(req_id)
        w = self.workers[wid]
        w.pending_requests = max(0, w.pending_requests - 1); w.pending_tokens = max(0, w.pending_tokens - req.cost)
        if seconds and seconds > 0:
            obs = req.cost / seconds
            w.speed = obs if not w.speed_observed else (1 - self.alpha) * w.speed + self.alpha * obs
            w.speed_observed = True

    def mark_dead(self, worker_id: int) -> list[Request]:
        """Mark a worker dead and return its in-flight requests that may still be retried elsewhere."""
        w = self.workers[worker_id]; w.alive = False
        lost = [r for (wid, r) in self.inflight.values() if wid == worker_id]
        for r in lost:
            self.inflight.pop(r.id, None)
        w.pending_requests = 0; w.pending_tokens = 0
        return [r for r in lost if r.attempts < self.max_attempts]

    def revive(self, worker_id: int) -> None:
        w = self.workers[worker_id]; w.alive = True; w.ready = True
