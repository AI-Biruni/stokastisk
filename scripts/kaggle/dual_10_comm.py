"""Stage 10 : host-staged transport (torch + native C++) vs NCCL. Experimental: a slower transport stays experimental."""
import json, os, pathlib, subprocess, sys
from _common import OUT, check, finish, run_guarded, save, skipped, versions, warn


def main() -> int:
    import torch, stokastisk
    from stokastisk.benchmarks import communication as C
    R = []
    check(R, "2 CUDA devices", torch.cuda.device_count() == 2, torch.cuda.device_count())
    host = C.bench_host_staged(0, 1)
    save("10_host_staged_torch.json", host)
    check(R, "host-staged (torch) payload byte-exact at every size", all(v["payload_verified"] for v in host.values()), {k: v["payload_verified"] for k, v in host.items()})
    # native C++
    native = None
    here = pathlib.Path(__file__).resolve().parent
    src = next((p for p in (here.parent.parent / "native/host_staged_xfer.cpp", here.parent / "native/host_staged_xfer.cpp") if p.is_file()), None)
    if src:
        from stokastisk.communication.native import NativeHostStaged, build_native, load_native
        lib_path = build_native(src, OUT / "native")
        if lib_path:
            lib = load_native(lib_path)
            native = C.bench_host_staged(0, 1, native=lambda s, d, chunk_bytes, n_buffers: NativeHostStaged(s, d, chunk_bytes=chunk_bytes, n_buffers=n_buffers, lib=lib))
            save("10_host_staged_native.json", native)
            check(R, "native C++ transport payload byte-exact at every size", all(v["payload_verified"] for v in native.values()))
        else:
            skipped(R, "native C++ transport", "nvcc build failed or nvcc missing")
    else:
        skipped(R, "native C++ transport", "native/host_staged_xfer.cpp not in this bundle")
    # NCCL reference
    env = dict(os.environ, STOKASTISK_OUT=str(OUT))
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", str(here / "dual_10_nccl_worker.py")],
                       capture_output=True, text=True, env=env, timeout=600)
    (OUT / "10_nccl_torchrun.log").write_text(r.stdout + r.stderr)
    check(R, "NCCL reference run exited 0", r.returncode == 0, r.stderr[-300:])
    cmp_ = {}
    if r.returncode == 0:
        nccl = json.loads((OUT / "10_nccl_rank1.json").read_text())["results"]
        cmp_ = {"torch_host_staged_vs_nccl": C.compare(host, nccl)}
        if native:
            cmp_["native_host_staged_vs_nccl"] = C.compare(native, nccl)
        save("10_comparison.json", cmp_)
        for k, v in cmp_.items():
            if v["host_staged_wins_at"]:
                warn(R, f"{k}: host-staged beats NCCL at {v['host_staged_wins_at']}", "investigate before any integration")
    finish("10_comm", R, {"versions": versions(), "comparison": cmp_, "integration_decision": "none: NCCL remains the only transport used by vLLM; host-staged stays experimental unless it wins and is separately validated"})
    return 0


if __name__ == "__main__":
    run_guarded("10_comm", main)
