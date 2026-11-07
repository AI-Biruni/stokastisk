"""TP=1 vs TP=2 on an identical workload."""
from __future__ import annotations

from ..config import StokastiskConfig
from .baseline import Workload, run_engine


def config_for(tp: int, **kw) -> StokastiskConfig:
    return StokastiskConfig(tensor_parallel_size=tp, execution_strategy="single_gpu" if tp == 1 else "tensor_parallel", **kw)


def run(model: str, workloads, tp: int, **kw):
    return run_engine(config_for(tp, **kw), model, workloads, label=f"tp{tp}")
