"""ReplicaPool: independent TP=1 vLLM engines, one per GPU, behind a deterministic token-aware scheduler.

Rules enforced here: the parent never initialises CUDA; each worker gets exactly one distinct physical device and must report a
distinct GPU UUID before the pool is considered ready; scripts that create a pool need an ``if __name__ == "__main__":`` guard (spawn)."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue
import threading
import time
from typing import Any

from ..config import StokastiskConfig
from ..errors import ConfigurationError, ReplicaError, UnsupportedParallelismError
from .queue import Request
from .routing import WorkerLoad
from .scheduler import Scheduler
from .worker import worker_main

log = logging.getLogger("stokastisk")


class ReplicaPool:
    def __init__(self, model: str, devices: list[int] | tuple[int, ...] = (0, 1), *, config: StokastiskConfig | None = None,
                 ready_timeout: float = 900.0, max_attempts: int = 2, engine_kwargs: dict[str, Any] | None = None) -> None:
        devices = tuple(devices)
        if not devices or len(set(devices)) != len(devices):
            raise UnsupportedParallelismError(f"devices must be distinct GPU indices, got {devices!r}")
        base = config or StokastiskConfig()
        if base.tensor_parallel_size != 1 or base.pipeline_parallel_size != 1:
            raise UnsupportedParallelismError("each replica is a TP=1/PP=1 engine; configure TP/PP = 1 (replicas are the parallelism)")
        self.config = StokastiskConfig(**{**base.as_dict(), "execution_strategy": "replica_parallel", "replica_devices": devices})
        self.model, self.devices, self._engine_kwargs = model, devices, dict(engine_kwargs or {})
        self._ctx = mp.get_context("spawn")
        self._out = self._ctx.Queue(); self._in: dict[int, Any] = {}; self._procs: dict[int, Any] = {}
        self.info: dict[int, dict[str, Any]] = {}
        self.sched = Scheduler([WorkerLoad(id=i, device=d, ready=False) for i, d in enumerate(devices)], max_attempts=max_attempts)
        self._next_id = 0; self._closed = False
        self._lock = threading.Lock()      # generate() calls share one result queue: they are serialised, never interleaved
        self.last_stats: dict[str, Any] = {}
        self.config.apply_environment()
        try:
            for i, d in enumerate(devices):
                self._spawn(i)
            self._await_ready(set(range(len(devices))), ready_timeout)
            self._check_isolation()
        except BaseException:
            self.close(); raise

    # ------------------------------------------------------------------ lifecycle
    def _spawn(self, i: int) -> None:
        d = self.devices[i]
        self._in[i] = self._ctx.Queue()
        p = self._ctx.Process(target=worker_main, name=f"stokastisk-replica-{i}-gpu{d}", daemon=False,
                              args=(i, d, self.model, self.config.as_dict(), self._engine_kwargs, self._in[i], self._out))
        p.start(); self._procs[i] = p

    def _await_ready(self, pending: set[int], timeout: float) -> None:
        end = time.time() + timeout
        while pending:
            try:
                msg = self._out.get(timeout=2.0)
            except _queue.Empty:
                for i in list(pending):
                    if not self._procs[i].is_alive():
                        raise ReplicaError(f"replica {i} (GPU {self.devices[i]}) exited before becoming ready (exitcode {self._procs[i].exitcode})")
                if time.time() > end:
                    raise ReplicaError(f"replicas {sorted(pending)} not ready within {timeout}s")
                continue
            if msg[0] == "ready":
                self.info[msg[1]] = msg[2]; self.sched.workers[msg[1]].ready = True; pending.discard(msg[1])
            elif msg[0] == "failed":
                raise ReplicaError(f"replica {msg[1]} failed to start: {msg[2]}\n{msg[3]}")

    def _check_isolation(self) -> None:
        uuids = [v.get("uuid") for v in self.info.values()]
        if None not in uuids and len(set(uuids)) != len(uuids):
            raise ReplicaError(f"replicas share a GPU (UUIDs {uuids}); refusing to continue")
        pids = [v["pid"] for v in self.info.values()]
        if len(set(pids)) != len(pids):
            raise ReplicaError("replicas share a process")

    def restart_dead(self, timeout: float = 900.0) -> list[int]:
        """Restart every worker that has exited. Never called implicitly."""
        dead = [i for i, p in self._procs.items() if not p.is_alive()]
        for i in dead:
            self.sched.workers[i].alive = True; self.sched.workers[i].ready = False; self._spawn(i)
        if dead:
            self._await_ready(set(dead), timeout); self._check_isolation()
        return dead

    # ------------------------------------------------------------------ inference
    def generate(self, prompts: list[str], *, max_tokens: int = 64, temperature: float = 0.0, timeout: float = 1800.0) -> list[str]:
        if self._closed:
            raise ReplicaError("pool is closed")
        if isinstance(prompts, str) or not isinstance(prompts, (list, tuple)) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a list/tuple of str")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive int")
        if not prompts:
            return []
        with self._lock:
            return self._generate_locked(list(prompts), max_tokens, temperature, timeout)

    def _generate_locked(self, prompts: list[str], max_tokens: int, temperature: float, timeout: float) -> list[str]:
        if self._closed:
            raise ReplicaError("pool is closed")
        t0 = time.time()
        reqs = [Request(id=self._next_id + k, prompt=p, max_tokens=max_tokens) for k, p in enumerate(prompts)]
        self._next_id += len(reqs)
        by_id = {r.id: r for r in reqs}
        results: dict[int, tuple[str, int]] = {}
        per_worker_seconds: dict[int, float] = {}
        todo = list(reqs); batch_no = 0; outstanding: dict[int, tuple[int, list[Request]]] = {}
        while todo or outstanding:
            if todo:
                plan = self.sched.assign_batch(todo); todo = []
                for wid, rs in plan.items():
                    if wid == -1:
                        raise ReplicaError("no available replica for the pending requests")
                    batch_no += 1; outstanding[batch_no] = (wid, rs)
                    self._in[wid].put(("generate", batch_no, [(r.id, r.prompt, r.max_tokens) for r in rs], temperature))
            try:
                msg = self._out.get(timeout=1.0)
            except _queue.Empty:
                for b, (wid, rs) in list(outstanding.items()):
                    if not self._procs[wid].is_alive():
                        del outstanding[b]; retry = self.sched.mark_dead(wid)
                        lost_ids = {r.id for r in rs}; retried = {r.id for r in retry}
                        if lost_ids - retried:
                            raise ReplicaError(f"replica {wid} died and requests {sorted(lost_ids - retried)} exhausted their retries")
                        todo.extend(retry)
                if time.time() - t0 > timeout:
                    raise ReplicaError("generate timed out")
                continue
            if msg[0] == "result":
                _, wid, b, res, stats = msg
                if b not in outstanding:
                    continue
                _, rs = outstanding.pop(b); per_worker_seconds[wid] = per_worker_seconds.get(wid, 0.0) + stats["seconds"]
                results.update(res)
                for r in rs:
                    self.sched.complete(r.id, seconds=stats["seconds"] * r.cost / max(1, sum(x.cost for x in rs)))
                self.sched.workers[wid].free_vram_mib = (None if stats.get("memory_total_mib") is None else stats["memory_total_mib"] - stats["memory_used_mib"])
            elif msg[0] in ("batch_error", "failed"):
                raise ReplicaError(f"replica {msg[1]} error: {msg[2]}\n{msg[3]}")
        wall = time.time() - t0
        out_tokens = sum(n for _, n in results.values())
        self.last_stats = {"wall_seconds": wall, "requests": len(reqs), "output_tokens": out_tokens, "output_tok_per_s": out_tokens / wall if wall else None,
                           "per_worker_busy_seconds": per_worker_seconds,
                           "assignment": {wid: sum(1 for d in self.sched.log if d.worker_id == wid and d.request_id in by_id) for wid in self.sched.workers}}
        return [results[r.id][0] for r in reqs]

    # ------------------------------------------------------------------ shutdown
    def close(self, timeout: float = 60.0) -> dict[str, Any]:
        if self._closed:
            return getattr(self, "close_report", {})
        self._closed = True
        for i, q in self._in.items():
            if self._procs[i].is_alive():
                try:
                    q.put(("shutdown",))
                except (ValueError, OSError):
                    log.exception("could not send shutdown to replica %s; it will be terminated", i)
        end = time.time() + timeout; report = {}
        for i, p in self._procs.items():
            p.join(max(0.0, end - time.time()))
            if p.is_alive():
                p.terminate(); p.join(10.0); report[i] = "terminated"
            else:
                report[i] = f"exit {p.exitcode}"
        for q in (*self._in.values(), self._out):
            try:
                q.close(); q.cancel_join_thread()
            except (ValueError, OSError):
                log.debug("queue already closed", exc_info=True)
        self.close_report = report
        return report

    def __enter__(self) -> "ReplicaPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
