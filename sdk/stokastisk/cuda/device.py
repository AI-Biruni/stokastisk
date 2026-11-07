"""Thin, optional CUDA device helpers on top of PyTorch. They NEVER touch vLLM's internal streams or contexts."""
from __future__ import annotations

from contextlib import contextmanager


def device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


@contextmanager
def on_device(index: int):
    import torch
    with torch.cuda.device(index):
        yield


def synchronize(index: int | None = None) -> None:
    import torch
    torch.cuda.synchronize(index)
