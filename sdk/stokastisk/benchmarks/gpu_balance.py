"""Compute / communication / idle accounting and imbalance for replica runs."""
from __future__ import annotations

from typing import Any

from ..telemetry import WorkerMetrics, balance_report


def from_replica_record(rec: dict[str, Any], workload: str) -> dict[str, Any]:
    r0 = rec["results"][workload]
    runs = r0["per_worker_busy_seconds"]; walls = r0["batch_latency_seconds"]["runs"]
    per = r0["gpu_telemetry"].get("per_gpu", {})
    ws = []
    for wid in sorted({w for r in runs for w in r}):
        m = WorkerMetrics(worker=f"replica{wid}", wall_s=sum(walls), compute_s=sum(r.get(wid, 0.0) for r in runs), comm_wait_s=0.0)
        dev = rec["workers"][wid]["device"] if wid in rec["workers"] else None
        if dev in per and "gpu_util_pct" in per[dev]:
            m.gpu_util_pct_mean = per[dev]["gpu_util_pct"]["mean"]
        ws.append(m)
    if len(ws) != 2:
        return {"status": "NOT_APPLICABLE", "reason": f"balance needs exactly 2 workers, got {len(ws)}"}
    return {"status": "MEASURED", **balance_report(ws[0], ws[1]),
            "note": "replicas have no inter-GPU communication (comm_wait_s = 0 by construction); compute = worker-reported generate() time"}
