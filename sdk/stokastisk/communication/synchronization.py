"""Buffer ownership tracking for the host-staged pipeline.

A pinned buffer is OWNED by exactly one stage at a time: free -> D2H (source stream) -> H2D (destination stream) -> free.
Each hand-over is a CUDA event; a stage must wait on the previous owner's event before touching the buffer."""
from __future__ import annotations


class BufferTicket:
    __slots__ = ("d2h_done", "h2d_done")

    def __init__(self) -> None:
        self.d2h_done = None     # recorded on the source stream after the device->host copy
        self.h2d_done = None     # recorded on the destination stream after the host->device copy
