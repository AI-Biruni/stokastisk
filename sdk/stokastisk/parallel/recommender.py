"""Rule-based, deterministic parallelism recommendation. It only RECOMMENDS: the caller decides (``Recommendation.to_config()``)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import ConfigurationError
from .memory import CUDA_CONTEXT_OVERHEAD_MIB, ModelSpec, load_model_spec

WORKLOADS = ("latency", "throughput")


@dataclass
class Recommendation:
    strategy: str                    # single_gpu | tensor_parallel | pipeline_parallel | replica_parallel | insufficient_memory | unsupported_architecture | no_gpu
    tensor_parallel_size: int
    pipeline_parallel_size: int
    reason: str
    warnings: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    experimental: bool = False
    estimates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_config(self, **kw):
        """A validated StokastiskConfig for this recommendation. Raises for non-runnable outcomes."""
        from ..config import StokastiskConfig
        if self.strategy not in ("single_gpu", "tensor_parallel", "pipeline_parallel", "replica_parallel"):
            raise ConfigurationError(f"no runnable configuration: {self.strategy}: {self.reason}")
        return StokastiskConfig(tensor_parallel_size=self.tensor_parallel_size, pipeline_parallel_size=self.pipeline_parallel_size,
                                execution_strategy=self.strategy, **kw)


def default_recommendation(topology) -> str:
    """Model-independent summary used by ``Topology.recommended_parallelism``."""
    n = topology.gpu_count
    if n == 0:
        return "no_gpu"
    if n == 1:
        return "single_gpu (TP=1)"
    fast = bool(topology.nvlink_available) or topology.peer_access.all_pairs_available
    if fast:
        return "single_gpu (TP=1) if the model fits one GPU; tensor_parallel (TP=2) is viable beyond one GPU's memory (fast interconnect measured)"
    return ("single_gpu (TP=1) if the model fits one GPU (replica_parallel for aggregate throughput); "
            "tensor_parallel (TP=2) only when the model does not fit one GPU: no NVLink / no CUDA peer access measured")


def recommend_parallelism(model, topology=None, workload: str = "latency", *, max_model_len: int = 4096, gpu_memory_utilization: float = 0.88,
                          min_concurrent_seqs: int = 1) -> Recommendation:
    if workload not in WORKLOADS:
        raise ConfigurationError(f"workload must be one of {WORKLOADS}")
    if topology is None:
        from ..topology import discover_topology
        topology = discover_topology(with_recommendation=False)
    spec = load_model_spec(model)
    n = topology.gpu_count
    w = list(topology.notes)
    if n == 0:
        return Recommendation("no_gpu", 1, 1, "no CUDA GPU detected", w)
    if spec.unsupported_reason:
        return Recommendation("unsupported_architecture", 1, 1, spec.unsupported_reason, w, estimates={"model": spec.to_dict()})
    vram = min((d.total_vram_mib for d in topology.devices if d.total_vram_mib), default=None)
    if vram is None:
        return Recommendation("single_gpu", 1, 1, "VRAM could not be measured; defaulting to TP=1", w + ["VRAM unavailable: fit not checked"])
    usable = vram * gpu_memory_utilization - CUDA_CONTEXT_OVERHEAD_MIB
    kv = spec.kv_mib(max_model_len * max(1, min_concurrent_seqs))
    need1 = spec.weight_mib + kv
    need2 = spec.weight_mib / 2 + kv / 2           # TP shards KV heads; PP shards layers: both roughly halve per-GPU KV
    est = {"model": spec.to_dict(), "usable_mib_per_gpu": round(usable, 1), "need_one_gpu_mib": round(need1, 1), "need_per_gpu_two_gpus_mib": round(need2, 1),
           "max_model_len": max_model_len, "note": "ESTIMATE from config.json; measure on the target"}
    fits1, fits2 = need1 <= usable, n >= 2 and need2 <= usable
    fast = bool(topology.nvlink_available) or topology.peer_access.all_pairs_available
    link = topology.pair_link(0, 1) if n >= 2 else None
    if not fits1 and not fits2:
        return Recommendation("insufficient_memory", 1, 1, f"needs ~{need1:.0f} MiB on one GPU (~{need2:.0f} MiB per GPU on two); usable per GPU ~{usable:.0f} MiB", w, estimates=est)
    if fits1:
        alts: list[dict[str, Any]] = []
        if n >= 2:
            alts.append({"strategy": "tensor_parallel", "tensor_parallel_size": 2, "note": "adds NCCL traffic"
                         + ("" if fast else f" over a link measured as {link} with no CUDA peer access"), "measured_on_rc6": "0.55x of TP=1 decode on Kaggle 2xT4 (Qwen2.5-1.5B)"})
            alts.append({"strategy": "pipeline_parallel", "pipeline_parallel_size": 2, "experimental": True, "note": "unmeasured on this hardware until Kaggle notebook 08 runs"})
        if workload == "throughput" and n >= 2:
            ram_ok = topology.system_ram_available_gib is None or topology.system_ram_available_gib > 2 * spec.weight_mib / 1024 + 2
            ww = ["replica_parallel is experimental: not Kaggle-verified until notebook 09 passes; speed-up over one GPU is unmeasured"]
            if not ram_ok:
                ww.append("host RAM may be tight while two replicas load the weights concurrently")
            return Recommendation("replica_parallel", 1, 1, "the complete model fits on each GPU and the goal is aggregate throughput: independent TP=1 engines avoid all inter-GPU traffic",
                                  w + ww, alts + [{"strategy": "single_gpu", "tensor_parallel_size": 1}], True, est)
        return Recommendation("single_gpu", 1, 1, "the model fits on one GPU: TP=1 has no inter-GPU communication" + ("" if fast else f" (the two GPUs are {link}, no NVLink/CUDA P2P)"), w, alts, False, est)
    ww = w + ["the model does not fit one GPU, so two GPUs are required"]
    if not fast:
        ww.append("no NVLink / CUDA peer access: expect communication-bound tensor parallelism (RC6 measured TP=2 at 0.55x TP=1 for a model that fits one GPU)")
    return Recommendation("tensor_parallel", 2, 1, f"needs ~{need1:.0f} MiB on one GPU but ~{need2:.0f} MiB per GPU on two (usable ~{usable:.0f} MiB): TP=2 is the verified two-GPU path",
                          ww, [{"strategy": "pipeline_parallel", "pipeline_parallel_size": 2, "experimental": True,
                                "note": "moves one activation tensor per stage boundary instead of an all-reduce per layer, but is unmeasured here until notebook 08 runs"}], False, est)
