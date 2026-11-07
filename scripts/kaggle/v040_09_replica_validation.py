"""Stage 09 (0.4.0): two independent TP=1 replicas. usage: v040_09_replica_validation.py [MODEL]"""
import sys, time
from _common import check, finish, gpu_used_mib, live_engine_children, run_guarded, versions, wait_children_gone


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    import stokastisk
    from stokastisk.parallel import ReplicaPool
    R = []
    check(R, "parent process has NOT initialised CUDA before spawning workers", not __import__("torch").cuda.is_initialized())
    before = gpu_used_mib()
    t0 = time.time()
    pool = ReplicaPool(model, devices=[0, 1], config=stokastisk.StokastiskConfig(max_model_len=2048))
    extra = {"versions": versions(), "load_seconds": time.time() - t0, "workers": pool.info, "gpu_before": before}
    try:
        info = pool.info
        check(R, "two workers ready", len(info) == 2)
        check(R, "distinct GPU UUIDs", len({v["uuid"] for v in info.values()}) == 2 and None not in {v["uuid"] for v in info.values()}, [v["uuid"] for v in info.values()])
        check(R, "distinct worker PIDs", len({v["pid"] for v in info.values()}) == 2)
        check(R, "each worker pinned to its own CUDA_VISIBLE_DEVICES", sorted(v["cuda_visible_devices"] for v in info.values()) == ["0", "1"])
        check(R, "both workers use TRITON_ATTN", all(v["backend"] == "TRITON_ATTN" for v in info.values()), [v["backend"] for v in info.values()])
        mem = gpu_used_mib() or []
        check(R, "independent memory allocation: BOTH GPUs hold a full model replica", len(mem) == 2 and all(x > 500 for x in mem), mem)
        prompts = ["Explain CUDA streams.", "Explain Tensor Core architecture.", "What is the capital of France?", "Name a prime number greater than 10."] * 4
        out = pool.generate(prompts, max_tokens=24)
        check(R, "16 concurrent requests all returned non-empty text, in order", len(out) == 16 and all(o.strip() for o in out))
        check(R, "identical prompts produced identical text (determinism across replicas)", all(out[i] == out[i % 4] for i in range(16)))
        st = pool.last_stats; extra["last_stats"] = st
        check(R, "both replicas received work", all(v > 0 for v in st["assignment"].values()), st["assignment"])
        out2 = pool.generate(["The capital of France is"], max_tokens=16)
        check(R, "single request works", bool(out2[0].strip()), out2[0][:60])
        check(R, "paris appears for the France prompts", any("paris" in o.lower() for o in out))
    finally:
        extra["close_report"] = pool.close()
    time.sleep(2); live, ignored, waited = wait_children_gone(timeout=60)
    after = gpu_used_mib() or []
    check(R, "all worker processes exited", all(s == "exit 0" for s in extra["close_report"].values()), extra["close_report"])
    check(R, "no live engine process after close", not live, live)
    check(R, "GPU memory released after close (<500 MiB each)", all(x < 500 for x in after), after)
    finish("09_replica_validation", R, {**extra, "gpu_after_close": after, "shutdown": {"live": live, "ignored": ignored, "waited_s": waited}})
    return 0


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    run_guarded("09_replica_validation", main)
