"""PP=2 (experimental). Stage-level compute/communication times are NOT exposed by vLLM's public API: they are reported as NOT_MEASURED
and never inferred. The GPU-utilisation busy proxy below is a hint about balance, not a measurement of stage time."""
from __future__ import annotations

from ..config import StokastiskConfig
from ..telemetry import imbalance_percent
from .baseline import Workload, run_engine


def config_for(**kw) -> StokastiskConfig:
    return StokastiskConfig(tensor_parallel_size=1, pipeline_parallel_size=2, execution_strategy="pipeline_parallel", **kw)


def run(model: str, workloads, **kw):
    rec = run_engine(config_for(**kw), model, workloads, label="pp2")
    rec["stage_timing"] = {"status": "NOT_MEASURED", "reason": "vLLM 0.25.1 does not expose per-stage compute/communication time through LLM; Nsight would be needed"}
    for name, r in rec["results"].items():
        per = r["gpu_telemetry"].get("per_gpu", {})
        busy = [per[i]["gpu_util_pct"]["mean"] * r["full_run_seconds"]["mean"] / 100 for i in sorted(per) if "gpu_util_pct" in per[i]][:2]
        if len(busy) == 2:
            r["balance_proxy"] = {"kind": "PROXY (sampled GPU utilisation x run time), not stage time", "busy_s": busy, "imbalance_percent": imbalance_percent(*busy)}
    return rec
