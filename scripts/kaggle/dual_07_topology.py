"""Stage 07: measured GPU topology. Records facts; differences from the baseline-observed environment are WARNINGs, not failures."""
import json
from _common import OUT, check, finish, run_guarded, save, skipped, versions, warn


def main() -> int:
    import stokastisk
    from stokastisk.telemetry import nsight_status, sample_once
    R = []
    t = stokastisk.discover_topology(); save("07_topology.json", t.to_dict()); print(t)
    check(R, "exactly 2 GPUs visible", t.gpu_count == 2, t.gpu_count)
    check(R, "all GPUs are Tesla T4", all("T4" in (d.name or "") for d in t.devices), [d.name for d in t.devices])
    check(R, "all GPUs are SM75", all(d.compute_capability == (7, 5) for d in t.devices), [d.compute_capability for d in t.devices])
    uu = [d.uuid for d in t.devices]
    check(R, "GPU UUIDs measured and distinct", None not in uu and len(set(uu)) == len(uu), uu)
    check(R, "VRAM measured on every GPU", all(d.total_vram_mib for d in t.devices), [d.total_vram_mib for d in t.devices])
    pm = t.peer_access.matrix
    check(R, "CUDA peer access measured in both directions", t.gpu_count == 2 and pm.get(0, {}).get(1) is not None and pm.get(1, {}).get(0) is not None, pm)
    check(R, "NCCL available to torch", t.nccl_available is True, t.nccl_compiled_version)
    check(R, "link class determined (not UNKNOWN)", t.pair_link(0, 1) != "UNKNOWN", t.pair_link(0, 1))
    check(R, "recommendation is rule-based and present", bool(t.recommended_parallelism), t.recommended_parallelism)
    if t.pair_link(0, 1) != "PHB":
        warn(R, "link class differs from the baseline-observed PHB", t.pair_link(0, 1))
    if t.peer_access.any_pair_available:
        warn(R, "CUDA peer access is available here (baseline environment had none)", pm)
    if t.nvlink_available:
        warn(R, "NVLink reported (baseline environment had none)", "")
    s = sample_once(); check(R, "telemetry sample obtained (nvidia-smi)", s["source"] != "unavailable" and len(s["gpus"]) == t.gpu_count, s["source"])
    ns = nsight_status(); save("07_nsight.json", ns)
    for tool in ("nsys", "ncu"):
        if ns[tool]["status"] != "AVAILABLE":
            skipped(R, f"Nsight {tool}", ns[tool].get("reason", ""))
    finish("07_topology", R, {"versions": versions(), "peer_access": pm, "link_class": t.link_class, "nvlink": t.nvlink_available,
                              "nccl_compiled": t.nccl_compiled_version, "nccl_runtime_transport": "see 04 NCCL log (not observable from torch alone)"})
    return 0


if __name__ == "__main__":
    run_guarded("07_topology", main)
