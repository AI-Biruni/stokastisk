"""Transfer-size sweep: host-staged (torch / native C++) vs NCCL send/recv. NCCL runs inside a 2-rank torchrun job."""
from __future__ import annotations

import os
import resource
import time
from typing import Any

from ..telemetry import summarize

SIZES = {"1KiB": 1 << 10, "4KiB": 4 << 10, "64KiB": 64 << 10, "1MiB": 1 << 20, "4MiB": 4 << 20, "16MiB": 16 << 20, "64MiB": 64 << 20, "128MiB": 128 << 20}


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def bench_host_staged(src: int = 0, dst: int = 1, *, sizes=SIZES, iters: int = 20, warmup: int = 5, chunk_bytes: int = 4 << 20, n_buffers: int = 2, native=None) -> dict[str, Any]:
    """Verifies the payload (byte-exact) on every size, then times ``iters`` transfers. ``native`` = NativeHostStaged class to use C++ instead."""
    import torch
    cls = native or __import__("stokastisk.communication.cuda_transport", fromlist=["HostStagedTransport"]).HostStagedTransport
    tr = cls(src, dst, chunk_bytes=chunk_bytes, n_buffers=n_buffers); out = {}
    for label, n in sizes.items():
        a = torch.randint(0, 255, (n,), dtype=torch.uint8, device=f"cuda:{src}"); b = torch.zeros(n, dtype=torch.uint8, device=f"cuda:{dst}")
        runs, cpu0 = [], None
        for i in range(warmup + iters):
            if i == warmup:
                cpu0 = _cpu(); w0 = time.perf_counter()
            t = time.perf_counter(); tr.transfer(a, b); dt = time.perf_counter() - t
            if i >= warmup:
                runs.append(dt)
        wall = time.perf_counter() - w0
        ok = bool(torch.equal(a.cpu(), b.cpu()))
        out[label] = {"bytes": n, "latency_s": summarize(runs), "gbps_median": n / summarize(runs)["median"] / 1e9, "payload_verified": ok,
                      "cpu_utilisation_pct": (_cpu() - cpu0) / wall * 100, "host_pinned_bytes": getattr(getattr(tr, "ring", None), "host_bytes", n_buffers * chunk_bytes),
                      "process_maxrss_mib": _rss_mib()}
    return out


def bench_nccl(rank: int, *, sizes=SIZES, iters: int = 20, warmup: int = 5) -> dict[str, Any]:
    """Inside an initialised 2-rank NCCL group. Each rank returns its own view; rank 1 (receiver) completion time is the transfer latency."""
    from ..communication.nccl import time_p2p
    out = {}
    for label, n in sizes.items():
        cpu0 = _cpu(); w0 = time.perf_counter()
        r = time_p2p(n, rank, iters=iters, warmup=warmup); wall = time.perf_counter() - w0
        out[label] = {"bytes": n, "latency_s": summarize(r["runs_s"]), "gbps_median": n / summarize(r["runs_s"])["median"] / 1e9,
                      "cpu_utilisation_pct": (_cpu() - cpu0) / wall * 100, "process_maxrss_mib": _rss_mib()}
    return out


def compare(host_staged: dict, nccl: dict) -> dict[str, Any]:
    """Per size: ratio of median latencies and which transport wins. Decision rule: host-staged is integrated nowhere unless it wins."""
    rows = {}
    for k in host_staged:
        if k in nccl:
            h, n = host_staged[k]["latency_s"]["median"], nccl[k]["latency_s"]["median"]
            rows[k] = {"host_staged_over_nccl": h / n, "faster": "host_staged" if h < n else "nccl"}
    wins = [k for k, v in rows.items() if v["faster"] == "host_staged"]
    return {"per_size": rows, "host_staged_wins_at": wins,
            "verdict": "host_staged faster at some sizes: investigate before any integration" if wins else "NCCL is faster at every measured size: host-staged stays experimental"}
