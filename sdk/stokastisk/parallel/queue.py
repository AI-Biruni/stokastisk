"""Request records and a deterministic FIFO queue."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic input-token estimate (~4 characters per token). Replace with a tokenizer when exactness matters."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class Request:
    id: int
    prompt: str
    max_tokens: int = 64
    input_tokens: int = 0
    attempts: int = 0
    enqueued_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.input_tokens:
            self.input_tokens = estimate_tokens(self.prompt)

    @property
    def cost(self) -> int:
        """Token-aware work estimate: prompt tokens (prefill) + requested output tokens (decode)."""
        return self.input_tokens + self.max_tokens


class RequestQueue:
    def __init__(self, maxsize: int | None = None) -> None:
        self._q: deque[Request] = deque(); self.maxsize = maxsize

    def put(self, r: Request, *, front: bool = False) -> None:
        if self.maxsize is not None and len(self._q) >= self.maxsize:
            raise OverflowError("request queue is full")
        (self._q.appendleft if front else self._q.append)(r)

    def get(self) -> Request | None:
        return self._q.popleft() if self._q else None

    def __len__(self) -> int:
        return len(self._q)

    @property
    def pending_tokens(self) -> int:
        return sum(r.cost for r in self._q)
