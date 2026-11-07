"""Stream helpers. Create your own streams; do not enqueue onto vLLM's streams from outside vLLM."""
from __future__ import annotations


def create_stream(device: int, *, priority: int = 0):
    import torch
    return torch.cuda.Stream(device=device, priority=priority)


def stream_synchronize(stream) -> None:
    stream.synchronize()
