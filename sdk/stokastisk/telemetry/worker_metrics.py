"""Per-worker accounting used by the balance benchmark: compute vs communication vs idle, and queue wait."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .metrics import imbalance_percent


@dataclass
class WorkerMetrics:
    worker: str
    wall_s: float = 0.0
    compute_s: float = 0.0
    comm_wait_s: float = 0.0
    queue_wait_s: list[float] = field(default_factory=list)
    requests: int = 0
    output_tokens: int = 0
    mem_used_mib: float | None = None
    gpu_util_pct_mean: float | None = None

    @property
    def idle_s(self) -> float:
        return max(0.0, self.wall_s - self.compute_s - self.comm_wait_s)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "idle_s": self.idle_s}


def balance_report(w0: WorkerMetrics, w1: WorkerMetrics) -> dict[str, Any]:
    """Imbalance on compute time and on total busy time, reported separately (utilisation alone does not prove balance)."""
    return {"compute_imbalance_percent": imbalance_percent(w0.compute_s, w1.compute_s),
            "busy_imbalance_percent": imbalance_percent(w0.compute_s + w0.comm_wait_s, w1.compute_s + w1.comm_wait_s),
            "workers": [w0.to_dict(), w1.to_dict()]}
