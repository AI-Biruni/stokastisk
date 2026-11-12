"""One replica worker process: one physical GPU, one TP=1 vLLM engine, one full copy of the weights.

The device is pinned with CUDA_VISIBLE_DEVICES *before* anything can initialise CUDA, and the worker never initialises CUDA itself
(identity/memory come from nvidia-smi), so vLLM's own engine-core child is the only CUDA user inside the worker's process tree."""
from __future__ import annotations

import os
import subprocess
import time
import traceback
from typing import Any


def gpu_identity(physical_index: int) -> dict[str, Any]:
    """uuid + memory.used of a PHYSICAL GPU via nvidia-smi (independent of CUDA_VISIBLE_DEVICES, no CUDA context)."""
    try:
        r = subprocess.run(["nvidia-smi", "-i", str(physical_index), "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        u, used, total = [x.strip() for x in r.stdout.strip().split(",")]
        return {"uuid": u, "memory_used_mib": float(used), "memory_total_mib": float(total)}
    except Exception:  # noqa: BLE001
        return {"uuid": None, "memory_used_mib": None, "memory_total_mib": None}


def worker_main(worker_id: int, device: int, model: str, config_dict: dict, engine_kwargs: dict, in_q, out_q) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)           # first statement: nothing has touched CUDA yet
    os.environ["STOKASTISK_REPLICA_WORKER"] = str(worker_id)
    llm = None
    try:
        t0 = time.time()
        import stokastisk
        cfg = stokastisk.StokastiskConfig(**{**config_dict, "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                                             "execution_strategy": "single_gpu", "replica_devices": None})
        llm = stokastisk.LLM(model, config=cfg, **engine_kwargs)
        ident = gpu_identity(device)
        out_q.put(("ready", worker_id, {"device": device, "pid": os.getpid(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                                        "load_seconds": time.time() - t0, "backend": llm.selected_attention_backend(), **ident}))
        from vllm import SamplingParams
        while True:
            msg = in_q.get()
            kind = msg[0]
            if kind == "shutdown":
                break
            if kind == "ping":
                out_q.put(("pong", worker_id, gpu_identity(device))); continue
            if kind == "generate":
                _, batch_id, items, temperature = msg           # items: [(request_id, prompt, max_tokens)]
                t1 = time.time()
                try:
                    # one vLLM call per distinct max_tokens so each request gets its own limit while still batching
                    results: dict[int, tuple[str, int]] = {}
                    for mt in sorted({m for _, _, m in items}):
                        group = [(rid, p) for rid, p, m in items if m == mt]
                        gen = llm.generate([p for _, p in group], SamplingParams(max_tokens=mt, temperature=temperature))
                        for (rid, _), o in zip(group, gen):
                            results[rid] = (o.outputs[0].text, len(o.outputs[0].token_ids))
                    out_q.put(("result", worker_id, batch_id, results, {"seconds": time.time() - t1, **gpu_identity(device)}))
                except Exception as e:  # noqa: BLE001
                    out_q.put(("batch_error", worker_id, batch_id, f"{type(e).__name__}: {e}", traceback.format_exc()))
    except BaseException as e:  # noqa: BLE001 - the parent must always learn why a worker went away
        out_q.put(("failed", worker_id, f"{type(e).__name__}: {e}", traceback.format_exc()))
    finally:
        try:
            if llm is not None:
                llm.close()
        except Exception:  # noqa: BLE001 - reported to the parent, which logs it; the worker must still announce its exit
            out_q.put(("close_error", worker_id, f"{type(llm).__name__}.close failed", traceback.format_exc()))
        out_q.put(("exited", worker_id, os.getpid()))
