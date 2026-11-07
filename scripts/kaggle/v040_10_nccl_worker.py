"""Run under torchrun --nproc_per_node=2: NCCL send/recv timings per size. Rank 1 (receiver) result is the transfer latency."""
import json, os, sys
import torch, torch.distributed as dist

if __name__ == "__main__":
    rank = int(os.environ["RANK"]); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from stokastisk.benchmarks.communication import bench_nccl
    res = bench_nccl(rank)
    out = os.path.join(os.environ.get("STOKASTISK_OUT", "."), f"10_nccl_rank{rank}.json")
    json.dump({"rank": rank, "results": res, "nccl_runtime": ".".join(map(str, torch.cuda.nccl.version())) + " (compile-time)"}, open(out, "w"), indent=1)
    dist.barrier(); dist.destroy_process_group()
