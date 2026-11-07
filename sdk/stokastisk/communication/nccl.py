"""NCCL timing helpers for a torchrun/``torch.distributed`` process group (2 ranks). NCCL is the reference transport; it is never replaced."""
from __future__ import annotations

import time
from typing import Any


def time_p2p(nbytes: int, rank: int, *, iters: int = 20, warmup: int = 5) -> dict[str, Any]:
    """rank 0 sends ``nbytes`` to rank 1 (NCCL send/recv). Returns per-iteration seconds measured on this rank (barrier-aligned)."""
    import torch, torch.distributed as dist
    t = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    runs = []
    for i in range(warmup + iters):
        dist.barrier(); torch.cuda.synchronize(); t0 = time.perf_counter()
        if rank == 0:
            dist.send(t, dst=1)
        else:
            dist.recv(t, src=0)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i >= warmup:
            runs.append(dt)
    return {"bytes": nbytes, "runs_s": runs}


def time_allreduce(nbytes: int, *, iters: int = 20, warmup: int = 5) -> dict[str, Any]:
    import torch, torch.distributed as dist
    t = torch.zeros(max(1, nbytes // 2), dtype=torch.float16, device="cuda"); runs = []
    for i in range(warmup + iters):
        dist.barrier(); torch.cuda.synchronize(); t0 = time.perf_counter()
        dist.all_reduce(t); torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i >= warmup:
            runs.append(dt)
    return {"bytes": nbytes, "runs_s": runs}
