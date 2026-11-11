"""Stage 16 : replica routing after a warm-up. usage: dual_16_replica_balance.py [MODEL]
Regression check for the notebook-09 defect (every request routed to GPU 0 once its speed was learned). On the real two-GPU pool it
requires that BOTH replicas receive work at every batch size >= 2 across repeated rounds, and reports aggregate tokens/s per batch size.
Throughput versus TP=1 is reported by the benchmark stages, not asserted here: a speed-up is only claimed from measured numbers."""
import sys, time
from _common import check, finish, gpu_used_mib, run_guarded, versions, wait_children_gone

BATCHES = (2, 8, 16, 32, 64)
ROUNDS = 3


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    import stokastisk
    from stokastisk.parallel import ReplicaPool
    R = []; rows = []
    check(R, "parent process has NOT initialised CUDA before spawning workers", not __import__("torch").cuda.is_initialized())
    pool = ReplicaPool(model, devices=[0, 1], config=stokastisk.StokastiskConfig(max_model_len=2048))
    extra = {"versions": versions(), "workers": pool.info, "model": model}
    try:
        check(R, "two workers ready on distinct GPU UUIDs", len({v["uuid"] for v in pool.info.values()}) == 2)
        pool.generate(["warm-up"] * 2, max_tokens=8)                       # one request per worker: both learn their speed
        for b in BATCHES:
            for rnd in range(ROUNDS):
                prompts = [f"Write one sentence about the number {i}." for i in range(b)]
                out = pool.generate(prompts, max_tokens=32)
                st = pool.last_stats; a = {int(k): v for k, v in st["assignment"].items()}
                # last_stats counts this call's requests only; the sum must equal the batch
                tot = sum(a.values())
                rows.append({"batch": b, "round": rnd, "assignment": a, "wall_seconds": st["wall_seconds"], "output_tok_per_s": st["output_tok_per_s"],
                             "per_worker_busy_seconds": st["per_worker_busy_seconds"]})
                check(R, f"b{b} r{rnd}: all {b} requests answered, assignment sums to the batch", len(out) == b and all(o.strip() for o in out) and tot == b, a)
                check(R, f"b{b} r{rnd}: BOTH replicas received work", all(a.get(w, 0) > 0 for w in (0, 1)), a)
                lo, hi = min(a.values()), max(a.values())
                check(R, f"b{b} r{rnd}: split within one request plus 25% of the batch", hi - lo <= 1 + b // 4, a)
        extra["rows"] = rows
    finally:
        extra["close_report"] = pool.close()
    time.sleep(2); live, ignored, waited = wait_children_gone(timeout=60)
    after = gpu_used_mib() or []
    check(R, "all worker processes exited", all(s == "exit 0" for s in extra["close_report"].values()), extra["close_report"])
    check(R, "no live engine process after close", not live, live)
    check(R, "GPU memory released after close (<500 MiB each)", all(x < 500 for x in after), after)
    finish("16_replica_balance", R, {**extra, "gpu_after_close": after, "shutdown": {"live": live, "ignored": ignored, "waited_s": waited}})
    return 0


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    run_guarded("16_replica_balance", main)
