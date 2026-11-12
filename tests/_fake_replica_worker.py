"""Stand-in for stokastisk.parallel.worker.worker_main (same message protocol, no GPU). Importable from spawned children via sys.path."""
import os, time


def worker_main(worker_id, device, model, config_dict, engine_kwargs, in_q, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if os.environ.get("FAKE_FAIL_START") == str(worker_id):
        out_q.put(("failed", worker_id, "RuntimeError: boom", "tb")); return
    uuid = os.environ.get("FAKE_SAME_UUID") and "GPU-same" or f"GPU-fake-{device}"
    out_q.put(("ready", worker_id, {"device": device, "pid": os.getpid(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "load_seconds": 0.0,
                                    "backend": "TRITON_ATTN", "uuid": uuid, "memory_used_mib": 1000.0, "memory_total_mib": 15000.0}))
    served = 0
    while True:
        msg = in_q.get()
        if msg[0] == "shutdown":
            if os.environ.get("FAKE_CLOSE_ERROR") == str(worker_id):
                out_q.put(("close_error", worker_id, "LLM.close failed", "tb"))
            if os.environ.get("FAKE_HANG_ON_SHUTDOWN") == str(worker_id):
                time.sleep(120)                              # ignores the shutdown request: the pool must terminate it
            break
        if msg[0] == "generate":
            _, b, items, temp = msg
            if os.environ.get("FAKE_DIE_ON_FIRST") == str(worker_id) and served == 0:
                os._exit(3)                                  # simulated crash with a batch in flight
            served += 1
            out_q.put(("result", worker_id, b, {rid: (f"w{worker_id}:{p}", mt) for rid, p, mt in items},
                       {"seconds": 0.01 * len(items), "memory_used_mib": 1000.0, "memory_total_mib": 15000.0, "uuid": uuid}))
    out_q.put(("exited", worker_id, os.getpid()))
