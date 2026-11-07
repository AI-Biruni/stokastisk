"""Memory helpers: free/total per device, pinned allocation, timed async copies."""
from __future__ import annotations

from typing import Any


def mem_info(device: int) -> dict[str, float]:
    import torch
    with torch.cuda.device(device):
        free, total = torch.cuda.mem_get_info()
    return {"free_mib": free / 2**20, "total_mib": total / 2**20, "used_mib": (total - free) / 2**20}


def pinned_empty(nbytes: int):
    import torch
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def timed_copy(dst, src, stream) -> dict[str, Any]:
    """Async copy on ``stream`` timed with CUDA events on the stream's device."""
    import torch
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        a.record(stream); dst.copy_(src, non_blocking=True); b.record(stream)
    b.synchronize()
    return {"bytes": src.numel() * src.element_size(), "ms": a.elapsed_time(b)}
