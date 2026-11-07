"""Independent TP=1 replicas vs one engine. Offline batch generation reports BATCH completion latency (individual request completion
times are not observable through vLLM's offline API)."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from ..config import StokastiskConfig
from ..telemetry import GpuSampler, provenance, summarize
from .baseline import Workload, _model_revision


def run(model: str, workloads, devices=(0, 1), **kw) -> dict[str, Any]:
    from ..parallel.replica import ReplicaPool
    wls = [workloads] if isinstance(workloads, Workload) else list(workloads)
    rec: dict[str, Any] = {"label": f"replicas{len(devices)}", "strategy": "replica_parallel", "model": model, "model_revision": _model_revision(model),
                           "provenance": provenance(), "devices": list(devices), "results": {}}
    t0 = time.perf_counter()
    pool = ReplicaPool(model, devices=list(devices), config=StokastiskConfig(**kw))
    rec["load_seconds"] = time.perf_counter() - t0; rec["workers"] = pool.info
    try:
        for w in wls:
            prompts = w.prompts()
            for _ in range(w.warmup):
                pool.generate(prompts, max_tokens=w.output_tokens, temperature=w.temperature)
            walls, toks, per_worker = [], [], []
            with GpuSampler(0.5) as gs:
                for _ in range(w.repeats):
                    pool.generate(prompts, max_tokens=w.output_tokens, temperature=w.temperature)
                    s = pool.last_stats; walls.append(s["wall_seconds"]); toks.append(s["output_tokens"]); per_worker.append(s["per_worker_busy_seconds"])
            rec["results"][w.name] = {"workload": asdict(w), "batch_latency_seconds": summarize(walls),
                                      "decode_tok_per_s": summarize([t / x for t, x in zip(toks, walls)]),   # aggregate output tok/s
                                      "per_worker_busy_seconds": per_worker, "assignment": pool.last_stats["assignment"],
                                      "gpu_telemetry": gs.summary(), "status": "MEASURED"}
        rec["status"] = "MEASURED"
    finally:
        rec["close_report"] = pool.close()
    return rec
