"""Stage 08 : PP=2 on the bundled vLLM wheel. usage: dual_08_pp_validation.py [MODEL]
A vLLM limitation or crash is reported as ERROR with the traceback + reason. It is NEVER converted to PASS."""
import os, pathlib, shutil, sys, time, traceback
from _common import OUT, check, finish, gpu_used_mib, process_tree, proc_detail, run_guarded, save, tree_loaded_libs, versions, wait_children_gone


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    cache = OUT / "triton-cache-pp2"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    from vllm import SamplingParams
    R = []
    stokastisk.validate_environment(require_gpus=2)
    before = gpu_used_mib()
    cfg = stokastisk.StokastiskConfig(tensor_parallel_size=1, pipeline_parallel_size=2, execution_strategy="pipeline_parallel", max_model_len=4096)
    check(R, "config: TP=1 PP=2 strategy pipeline_parallel, world size 2", (cfg.tensor_parallel_size, cfg.pipeline_parallel_size, cfg.world_size) == (1, 2, 2))
    try:
        stokastisk.StokastiskConfig(tensor_parallel_size=2, pipeline_parallel_size=2); ok = False
    except stokastisk.UnsupportedParallelismError:
        ok = True
    check(R, "config rejects TP=2 x PP=2 on two GPUs", ok)
    t0 = time.time()
    try:
        llm = stokastisk.LLM(model, config=cfg)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(); print(tb, file=sys.stderr)
        R.append({"check": "PP=2 engine initialisation on vLLM 0.25.1", "status": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:600]}"})
        finish("08_pp_validation", R, {"versions": versions(), "traceback": tb, "limitation": "vLLM/PP failed to initialise; PP=2 is NOT supported by this build until this is resolved"})
        return 1
    load_s = time.time() - t0
    try:
        check(R, "engine config backend == TRITON_ATTN", llm.selected_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
        pc = llm.engine.llm_engine.vllm_config.parallel_config
        check(R, "vLLM parallel_config reports TP=1 PP=2", (pc.tensor_parallel_size, pc.pipeline_parallel_size) == (1, 2), (pc.tensor_parallel_size, pc.pipeline_parallel_size))
        sp = SamplingParams(max_tokens=48, temperature=0.0)
        a = llm.generate(["The capital of France is"], sp)[0].outputs[0].text; b = llm.generate(["The capital of France is"], sp)[0].outputs[0].text
        check(R, "non-empty output", bool(a.strip()), a[:80]); check(R, "deterministic at temperature 0", a == b)
        peak = gpu_used_mib() or []
        check(R, "BOTH GPUs hold memory during PP generation (both stages active)", len(peak) == 2 and all(x > 500 for x in peak), peak)
        arts = sorted({p.name for p in cache.rglob("*") if p.is_file() and "kernel_unified_attention" in p.name})
        check(R, "Triton cache holds compiled kernel_unified_attention", bool(arts), arts[:3] or "none")
        libs = tree_loaded_libs(); flat = sorted({x for v in libs.values() if isinstance(v, list) for x in v})
        bad = [x for x in flat if any(k in x.lower() for k in ("vllm_flash_attn", "_vllm_fa", "flashinfer", "flashmla"))]
        check(R, "no Flash library mapped in any process", not bad, bad or f"{len(flat)} libs inspected")
        kids = [p for p in (process_tree() or []) if p]
        extra = {"load_seconds": load_s, "text": a, "gpu_used_MiB": {"before": before, "peak": peak}, "engine_processes": [proc_detail(p) for p in kids][:12]}
    finally:
        llm.close()
    time.sleep(2); live, ignored, waited = wait_children_gone(timeout=30)
    after = gpu_used_mib() or []
    check(R, "no live engine process after close", not live, live)
    check(R, "GPU memory released after close (<500 MiB each)", all(x < 500 for x in after), after)
    finish("08_pp_validation", R, {"versions": versions(), **extra, "gpu_after_close": after, "shutdown": {"live": live, "ignored": ignored, "waited_s": waited}})
    return 0


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    run_guarded("08_pp_validation", main)
