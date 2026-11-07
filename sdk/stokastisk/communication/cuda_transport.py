"""EXPERIMENTAL host-staged GPU->GPU transfer for platforms without CUDA peer access.

    src device --cudaMemcpyAsync--> pinned host buffer[b] --cudaMemcpyAsync--> dst device

Chunks cycle through ``n_buffers`` pinned buffers so chunk i+1's D2H overlaps chunk i's H2D. This is NOT an NVLink replacement, it
does not replace NCCL, and it is not used by the vLLM engine. Whether it beats NCCL is a measurement (see benchmarks/communication)."""
from __future__ import annotations

import time
from typing import Any

from .pinned_memory import PinnedBufferRing
from .synchronization import BufferTicket


class HostStagedTransport:
    def __init__(self, src_device: int, dst_device: int, *, chunk_bytes: int = 4 << 20, n_buffers: int = 2) -> None:
        import torch
        if src_device == dst_device:
            raise ValueError("source and destination must be different devices")
        self.src, self.dst, self.chunk, self.nbuf = src_device, dst_device, chunk_bytes, n_buffers
        self.ring = PinnedBufferRing(n_buffers, chunk_bytes)
        self.s_src = torch.cuda.Stream(device=src_device); self.s_dst = torch.cuda.Stream(device=dst_device)
        self.tickets = [BufferTicket() for _ in range(n_buffers)]

    def transfer(self, src_tensor, dst_tensor) -> dict[str, Any]:
        """Copy ``src_tensor`` (on src device) into ``dst_tensor`` (on dst device); returns timing. Blocks until complete."""
        import torch
        a = src_tensor.view(torch.uint8).reshape(-1); b = dst_tensor.view(torch.uint8).reshape(-1)
        if a.numel() != b.numel():
            raise ValueError("size mismatch")
        if a.device.index != self.src or b.device.index != self.dst:
            raise ValueError("tensors are not on the configured devices")
        n = a.numel(); t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        for t in self.tickets:
            t.d2h_done = t.h2d_done = None
        with torch.cuda.device(self.dst):
            ev0.record(self.s_dst)
        for i, off in enumerate(range(0, n, self.chunk)):
            ln = min(self.chunk, n - off); k = i % self.nbuf; tk = self.tickets[k]; buf = self.ring.buffers[k][:ln]
            if tk.h2d_done is not None:                       # buffer k still being read by the previous H2D: wait before overwriting it
                self.s_src.wait_event(tk.h2d_done)
            with torch.cuda.stream(self.s_src):
                buf.copy_(a[off:off + ln], non_blocking=True)
                tk.d2h_done = torch.cuda.Event(); tk.d2h_done.record(self.s_src)
            self.s_dst.wait_event(tk.d2h_done)                # H2D may start only after this chunk's D2H completed
            with torch.cuda.stream(self.s_dst):
                b[off:off + ln].copy_(buf, non_blocking=True)
                tk.h2d_done = torch.cuda.Event(); tk.h2d_done.record(self.s_dst)
        ev1.record(self.s_dst)
        self.s_dst.synchronize(); self.s_src.synchronize()
        wall = time.perf_counter() - t0
        return {"bytes": n, "wall_s": wall, "gpu_ms": ev0.elapsed_time(ev1), "gbps": n / wall / 1e9 if wall else None,
                "chunks": (n + self.chunk - 1) // self.chunk, "host_pinned_bytes": self.ring.host_bytes}
