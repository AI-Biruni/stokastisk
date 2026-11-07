"""Event helpers: dependency edges between streams (also across devices) and GPU-side timing."""
from __future__ import annotations


def new_event(*, timing: bool = False):
    import torch
    return torch.cuda.Event(enable_timing=timing)


def record(event, stream=None):
    event.record(stream); return event


def make_wait(stream, event) -> None:
    """cudaStreamWaitEvent: ``stream`` will not run later work until ``event`` completes (event may belong to another device)."""
    stream.wait_event(event)


def elapsed_ms(start, end) -> float:
    end.synchronize(); return start.elapsed_time(end)
