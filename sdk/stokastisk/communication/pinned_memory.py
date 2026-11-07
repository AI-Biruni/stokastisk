"""A small ring of page-locked host buffers (cudaHostAlloc via PyTorch's pinned allocator)."""
from __future__ import annotations


class PinnedBufferRing:
    def __init__(self, n_buffers: int, nbytes: int) -> None:
        if n_buffers < 1 or nbytes < 1:
            raise ValueError("need >=1 buffer of >=1 byte")
        import torch
        self.n, self.nbytes = n_buffers, nbytes
        self.buffers = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(n_buffers)]

    @property
    def host_bytes(self) -> int:
        return self.n * self.nbytes
