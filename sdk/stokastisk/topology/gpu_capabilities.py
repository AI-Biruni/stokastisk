"""Basic GPU properties from PyTorch (the primary provider). cuda-python is an optional, never-required refinement."""
from __future__ import annotations

from .models import DeviceInfo


def torch_devices() -> list[DeviceInfo]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            d = DeviceInfo(index=i, name=p.name, compute_capability=(p.major, p.minor), total_vram_mib=p.total_memory / 2**20,
                           sources={"name": "torch", "compute_capability": "torch", "total_vram_mib": "torch"})
            u = getattr(p, "uuid", None)
            if u is not None:
                d.uuid = f"GPU-{u}" if not str(u).startswith("GPU-") else str(u); d.sources["uuid"] = "torch"
            try:
                with torch.cuda.device(i):
                    free, total = torch.cuda.mem_get_info()
                d.free_vram_mib, d.used_vram_mib = free / 2**20, (total - free) / 2**20; d.sources["free_vram_mib"] = "torch"
            except Exception:  # noqa: BLE001 - mem_get_info creates a CUDA context; leave unset if that fails
                pass
            out.append(d)
        return out
    except Exception:  # noqa: BLE001
        return []


def cuda_python_version() -> str | None:
    try:
        from importlib import metadata
        return metadata.version("cuda-python")
    except Exception:  # noqa: BLE001
        return None
