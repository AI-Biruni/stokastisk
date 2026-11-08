"""Benchmark one mode over a workload set. usage: dual_bench.py MODE MODEL SET     MODE: tp1|tp2|pp2|replicas2   SET: standard|scaling
Writes bench_<MODE>_<SET>.json (status MEASURED or ERROR+traceback). All repeats are recorded; nothing is best-of."""
import json, sys, traceback
from _common import OUT, finish, gpu_used_mib, run_guarded, save, versions


def main() -> int:
    mode, model, wset = sys.argv[1], sys.argv[2], sys.argv[3]
    from stokastisk.benchmarks import baseline as B
    wls = {"standard": [B.SHORT, B.LONG], "scaling": B.SCALING}[wset]
    if wset == "scaling":                       # keep the 64-request run affordable on a 30 GiB / 4-core host
        for w in wls:
            w.repeats, w.warmup = 3, 1
    name = f"bench_{mode}_{wset}"
    rec = {"mode": mode, "model": model, "set": wset, "status": "ERROR", "gpu_used_before_MiB": gpu_used_mib()}
    try:
        if mode in ("tp1", "tp2"):
            from stokastisk.benchmarks import tensor_parallel as m; rec.update(m.run(model, wls, int(mode[2])))
        elif mode == "pp2":
            from stokastisk.benchmarks import pipeline_parallel as m; rec.update(m.run(model, wls))
        elif mode == "replicas2":
            from stokastisk.benchmarks import replica_parallel as m; rec.update(m.run(model, wls, devices=(0, 1)))
        else:
            raise SystemExit(f"unknown mode {mode}")
    except Exception as e:  # noqa: BLE001
        rec.update(status="ERROR", error=f"{type(e).__name__}: {str(e)[:800]}", traceback=traceback.format_exc()); print(rec["traceback"], file=sys.stderr)
    rec["gpu_used_after_MiB"] = gpu_used_mib()
    save(f"{name}.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("mode", "status", "error", "load_seconds")}, indent=1))
    return 0 if rec["status"] == "MEASURED" else 1


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    run_guarded("bench_driver", main)
